---
title: 【学习笔记】通过文字检索视频片段：VCED之后端逻辑
date: 2022-11-23 21:49:27
index_img: /img/datawhale.jpg
tags: ['VCED', 'multimodal']
mathjax: false
math: false
categories: 
- notes
---

本篇文章是跨模态检索工具 VCED 介绍的第五篇，前面介绍了前端部分，本篇来研究后端部分。

<!--more--->

## 一、概览

项目后端共由四个文件组成，其中 VideoLoader 负责对上传的视频进行关键帧提取，CustomClipText 负责将上传的图片转换为向量数据，CustomClipImage 负责将提取的关键帧转换为向量数据，SimpleIndexer 负责向量数据的检索。

- VideoLoader
- CustomClipText
- CustomClipImage
- SimpleIndexer

## 二、VideoLoader

VideoLoader 继承自 Executor，在整个后端中执行抽帧操作。

这里简单介绍下核心代码：

```py
# 将提取函数注册在路由的 /extract 里
@requests(on='/extract')
def extract(self, docs: DocumentArray, parameters: Dict, **kwargs):
    """
    Load the video from the Document.uri, extract frames and audio. The extracted data are stored in chunks.

    :param docs: the input Documents with either the video file name or data URI in the `uri` field
    :param parameters: A dictionary that contains parameters to control
        extractions and overrides default values.
    Possible values are `ffmpeg_audio_args`, `ffmpeg_video_args`, `librosa_load_args`. Check out more description in the `__init__()`.
    For example, `parameters={'ffmpeg_video_args': {'s': '512x320'}`.
    """
    for doc in docs:
        # 每个 doc 对应一段视频链接，先把它封装成一个对象
        with tempfile.TemporaryDirectory() as tmpdir:
            source_fn = (
                self._save_uri_to_tmp_file(doc.uri, tmpdir)
                if self._is_datauri(doc.uri)
                else doc.uri
            )

            # extract all the frames video
            if 'image' in self._modality:
                ffmpeg_video_args = deepcopy(self._ffmpeg_video_args)
                ffmpeg_video_args.update(parameters.get('ffmpeg_video_args', {}))
                # 调用内部函数，把视频帧抽出来变成 tensor
                frame_tensors = self._convert_video_uri_to_frames(
                    source_fn, doc.uri, ffmpeg_video_args
                )
                for idx, frame_tensor in enumerate(frame_tensors):
                    self.logger.debug(f'frame: {idx}')
                    chunk = Document(modality='image')
                    # chunk.blob = frame_tensor
                    max_size = 240
                    img = Image.fromarray(frame_tensor)
                    if img.size[0] > img.size[1]:
                        width = max_size
                        height = math.ceil(max_size / img.size[0] * img.size[1])
                    else:
                        height = max_size
                        width = math.ceil(max_size / img.size[1] * img.size[0])
                    img = img.resize((width, height))
                    chunk.tensor = np.asarray(img).astype('uint8')
                    print(chunk.tensor.shape)
                    # chunk.tensor = np.array(frame_tensor).astype('uint8')
                    chunk.location = (np.uint32(idx),)
                    chunk.tags['timestamp'] = idx / self._frame_fps
                    if self._copy_uri:
                        chunk.tags['video_uri'] = doc.uri
                    # 到这里视频帧都被存在 doc 的 chunk 里了。
                    doc.chunks.append(chunk)
```

VideoLoader 的子函数中，有专门负责提取视频帧的参数，来得到视频分辨率的；有获取音频采样率等信息的；也有获取字幕文本的（即尝试查找 srt字幕文件）。

## 三、CustomClipText

该模块负责调用 CLIP 来实现把文本变成 embeddings 的功能。它的关键算法是：

```py
def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
    print('clip_text encode')
    for docs_batch in DocumentArray(
        filter(
            lambda x: bool(x.text),
            docs[parameters.get('traversal_paths', self.traversal_paths)],
        )
    ).batch(batch_size=parameters.get('batch_size', self.batch_size)) :

        text_batch = docs_batch.texts
        t1 = time.time()
        with torch.inference_mode():
            input_tokens = [self.model.encode_text(clip.tokenize([t, "unknown"]).to(self.device)) for t in text_batch] # self._generate_input_tokens(text_batch)
            embeddings = input_tokens # self.model.get_text_features(**input_tokens).cpu().numpy()
            for doc, embedding in zip(docs_batch, embeddings):
                doc.embedding = embedding
                # doc.embedding = np.array(embedding).astype('float32')[0]
        t2 = time.time()
        print("encode text cost:", t2 - t1)
        print(t1)
        print(t2)
```

你可以看到，它实际上就是调用了一个模型，并对数据进行了预处理，最后对文本数据进行编码，以 DocumentArray 形式存储，便于后续传值。

## 四、CustomClipImage

与 CustomClipText 类似，通过 CLIP 模型把图片变成 Embedding。只要有 CLIP 模型，对数据进行与处理后即可得到它的 Embedding。

```py
def encode(self, docs: DocumentArray, parameters: dict, **kwargs):
    t1 = time.time()
    print('clip_image encode', t1)
    document_batches_generator =  DocumentArray(
        filter(
            lambda x: x is not None,
            docs[parameters.get('traversal_paths', self.traversal_paths)],
        )
    ).batch(batch_size=parameters.get('batch_size', self.batch_size))
    with torch.inference_mode():
        for batch_docs in document_batches_generator:
            print('in for')
            for d in batch_docs:
                print('in clip image d.uri', d.uri, len(d.chunks))
                # tensor = self._generate_input_features(tensors_batch)
                tensors_batch = []
                for c in d.chunks:
                    if (c.modality == 'image'):
                        image_embedding = self.model.encode_image(self.preprocessor(Image.fromarray(c.tensor)).unsqueeze(0).to(self.device))
                        # tensors_batch.append(image_embedding)
                        tensors_batch.append(np.array(image_embedding).astype('float32'))
                embedding = tensors_batch
                d.embedding = embedding
    t2 = time.time()
    print('clip_image encode end', t2 - t1, t2)
```

注意这里也将embedding 编码成了 DocArray形式进行存储。

## 五、SimpleIndexer

这里实现了关键的检索功能，首先需要将所有资源转化为他们的 Embedding（即 DocArray），然后使用向量相似度比较的方式进行检索。检索的代码如下：

```py
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )

        traversal_right = parameters.get(
            'traversal_right', self.default_traversal_right
        )
        traversal_left = parameters.get('traversal_left', self.default_traversal_left)
        match_args = SimpleIndexer._filter_match_params(docs, match_args)
        # print('in indexer',docs[traversal_left].embeddings.shape, self._index[traversal_right])
        texts: DocumentArray = docs[traversal_left]
        stored_docs: DocumentArray = self._index[traversal_right]

        doc_ids = parameters.get("doc_ids")
        t1 = time.time()
        with torch.inference_mode():
            t1_00 = time.time()
            for text in texts:
                result = []
                text_features = text.embedding
                text.embedding = None
                for sd in stored_docs:
                    if doc_ids is not None and sd.uri not in doc_ids:
                        continue
                    images_features = sd.embedding
                    print('images len',len(images_features))
                    t1_0 = time.time()
                    tensor_images_features = [Tensor(image_features) for image_features in images_features]
                    t1_1 = time.time()
                    for i, image_features in enumerate(tensor_images_features):
                        tensor = image_features
                        probs = self.score(tensor, text_features)
                        result.append({
                            "score": probs[0][0],
                            "index": i,
                            "uri": sd.uri,
                            "id": sd.id
                        })
                    t1_2 = time.time()
                    print("tensor cost:", t1_1 - t1_0)
                    print("part score cost:", t1_2 - t1_1)
                    print(t1_0)
                    print(t1_1)
                    print(t1_2)
                t2 = time.time()
                print('score cost:', t2 - t1)
                # print(parameters, type(parameters.get("thod")))
                index_list = self.getMultiRange(result,0.1 if parameters.get("thod") is None else parameters.get('thod'), parameters.get("maxCount"))
                t3 = time.time()
                print('range cost:', t3 - t2)
                print(t1)
                print(t1_00)
                print(t2)
                print(t3)
                # print(index_list)
                docArr = DocumentArray.empty(len(index_list))
                for i, doc in enumerate(docArr):
                    doc.tags["leftIndex"] = index_list[i]["leftIndex"]
                    doc.tags["rightIndex"] = index_list[i]["rightIndex"]
                    # print(index_list[i])
                    doc.tags["maxImageScore"] = float(index_list[i]["maxImage"]["score"])
                    doc.tags["uri"] = index_list[i]["maxImage"]["uri"]
                    doc.tags["maxIndex"] = index_list[i]["maxImage"]["index"]
                # print(docArr)
                text.matches = docArr
```

对于每段匹配的视频帧，还会在最终有一个打分排序的操作。打分的过程其实就是对特征进行标准化，按照余弦相似度计算，最终通过softmax模型得出probability。实际代码如下：

```py
    def score(self, image_features, text_features):

        logit_scale = self.model.logit_scale.exp()
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

        # print(" img Label probs:", probs)
        return probs
```

## 总结

总的来看 VCED 的后端部分不算复杂，只是忠实地使用了 Jina 提供的框架，将多模态数据抽象为 Document 后利用 Jina 提供的 api 进行处理继承 Executor 类，然后自定义自己的操作。

如果想要实现自己的 Executor，也可以类比上面提到的模块来继承一个；不过多数用户可以想到的功能都已经被上传到Jina Hub上，VideoLoader的主体也可以在hub中进行访问，可以直接调用封装好的Executor，实现自己的功能模块。