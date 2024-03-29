---
title: 【学习笔记】开始认识LLM和一些概念
date: 2023-11-15 00:24:10
index_img: /img/datawhale.jpg
tags: ['LLM', 'datawhale']
mathjax: false
math: false
categories: 
- notes
---

### 一、背景和大纲

做大模型相关应用有一阵子了，但一直没沉下心来总结常见的概念和技术，对于很多名词也仅仅是浅尝辄止。每天早上公司的聊天软件都会推送周围同事的热门技术文章或设计文档，顿时产生了技术焦虑。什么是 agent？什么是 langchain ？收藏夹里积灰的链接越来越多，但始终腾不出时间来消灭它们。终于在某天晚上我想通了，不给自己规定个Deadline，人是始终不会主动行动的。刚刚好 Datawhale 在集合一批小伙伴进行组队学习，我也年轻一把，和小朋友们一起体验组队打卡的乐趣。

我打算用两周时间，整理下大模型相关的技术名词和技术背景，辅以几个demo实践。不求深入，但求理解，这样一是提升沟通效率，二是扩展思路、开阔眼界，扫除自己的思维盲区。

根据课程大纲，我将从下面几个方面进行逐步介绍和学习。

1. 何为大模型、大模型特点是什么
2. LangChain 是什么
3. 大模型开发流程及架构
4. 如何调用大模型 API
5. 知识库文档的搭建
6. 如何设计一段靠谱的prompt
7. 大模型应用产品的迭代和评估
8. 常见的前后端开发架构，方便快速装逼的关键

### 二、初出茅庐

简单介绍一些常见的概念，避免技术焦虑。

#### 1. 什么是大模型、LLM，大模型的能力、特点有哪些，跟我们前几年说的预训练模型有什么区别？

大语言模型（英文：Large Language Model，缩写LLM），顾名思义，首先是个语言模型，其次参数量特别巨大，训练数据也特别多。

虽然看起来是废话，但其实很关键。
1）参数量大代表运行成本高、时延高，想要做一些即时处理任务会比较困难；
2）训练数据要求质量高而且多，这就带来极高的训练成本，不是个人开发者所能承受得了的；
3）大语言模型也是语言模型，这意味着模型是以理解人类语言为目的进行的训练，如果你硬要将他用在什么数字信号分类场景也不是不可以，但这完全违背了语言模型的能力。因此在实际使用中，需要考虑到应用场景。

大模型只是参数量变大、训练数据变多，其实架构也是传统的 transformer 架构。只不过，当参数量、训练数据量都真的变得非常大时，量变会带来质变，大模型会产生一种“智能涌现”现象，尤其在解决复杂任务时表现出了惊人的潜力。

**1、智能涌现**

所谓智能涌现，主要是指大模型相较于传统模型，诞生了以下三个能力：
1. 泛化能力，即指令遵循能力。大模型可以根据用户的指令和其他信息来应对多种任务，**而不需要额外的训练或者参数微调**。
LLM能够根据任务指令执行任务，而无需事先见过具体示例，以往的模型一般都是任务专用的，需要准备几千到几十万条领域相关的数据进行微调，才能在某个领域使用。但LLM的泛化能力已经强大到可以应对大部分的自然语言处理相关问题，而不需要单独给他预训练。训练一个大模型，分类、摘要、生成等任务都可以做了，只要你**在提问时提供足够的信息辅助大模型进行推理**就行。
2. 上下文学习能力。通过给LLM一些示例，LLM可以做到利用上下文进行快速学习。这种能力允许语言模型在提供自然语言指令或多个任务示例的情况下，通过理解上下文并生成相应输出的方式来执行任务，而无需额外的训练或参数更新。
3. 逐步推理过程。对于包含逻辑推理过程的复杂问题，普通模型一般难以解决；但大模型可以通过上下文学习，与用户不断交互，按照一定的推理过程来解决这些任务。这种能力被称之为“思维链”。

**2、基座模型**

大模型本身具备应对各式各样或简单或复杂的应用场景的能力，已经训练完毕的大模型可以直接赋能多种场景，可以作为一种基础设施，支持上游应用开发。重复造轮子在这个大模型时代不再流行，毕竟这个轮子已经太大了，远远不是个人开发者能够承担得起的。

**3、以对话作为统一的交互入口**

#### 2. 常见的大模型有哪些？

gpt-3.5-turbo是ChatGPT产品背后的LLM（之一）, gpt4, llama, chatglm, baichuan, 文心一言, 星火大模型等

其中 llama, chatglm, baichuan 开源。

#### 3. LangChain

LLM仍然存在能力上的局限和边界，如LLM不能很好的解决下面的问题：
1. 数据时效性。比如你想让 chatgpt 回答今天的天气，或者今天是周几，chatgpt大概率是不会回答正确的。
2. 幻觉，即大模型会一本正经的胡说八道。
3. 领域知识，即无法直接应用在某个小众领域，因为大模型没有这个领域的知识。
4. 多模态输入输出，即目前做不到输入输出可能为图像或视频。

大模型之所以没办法解决上面的问题，是因为大模型缺少这方面的知识。对于传统模型，一个很自然的想法就是使用领域知识进行微调，以期模型获得相应的知识；但大模型不用，也没有必要。我们只要在上下文里引入相关知识就好了。

1. 如果想要问天气，则大模型去请求查询天气预报数据库的插件，插件得到数据后返回给大模型，大模型整理成自然语言返回给用户；
2. 如果有幻觉，或者领域知识不足，则构建一个领域专属外挂知识库，大模型在回答前先利用插件检索知识库内的相关内容，将其作为上下文；再整理成自然语言，反馈给客户，可以极大避免幻觉的产生。
3. 要想拥有跨模态的输入输出能力，或者其他更复杂的能力，只需要实现相应的插件即可，大模型只需做最后的知识汇总。

一个使用插件的典型技术方案是 LangChain。Langchain是一个大语言模型的开发框架，是用来将 LLM 和 第三方插件以及外挂知识库集成起来的编程框架。

LangChain把大语言模型在线api的访问封装起来了，示例如下：

```py
import os
import configparser
from langchain.llm import OpenAI

config = configparser.ConfigParser()
config.read('dev.config')

# OpenAI提供的开发者Secret Key
os.environ['OPENAI_API_KEY'] = config['keys']['openai_api_key']

# 创建OpenAI的LLM，默认为text-davinci-003, temperature控制结果随机程度，取值[0, 1]，越大越随机。
llm = OpenAI(temperature = 0.1)

input_text = '肚子胀不消化，吃'

print(llm(input_text))
```

LangChain还把prompt给封装起来了，方便调用模板进行访问：

```py
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# HuggingFace提供的API TOKEN，即开发者Secret Key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['keys']['huggingfacehub_api_token']

# 创建Google T5的LLM
llm_flan_t5 = HuggingFaceHub(repo_id = 'google/flan-t5-xl', model_kwargs = {'temperature':1e-10})
# 创建自定义模板
template = """Question: {question}
   Answer: """
# 通过模板，创建prompt
prompt = PromptTemplate(template = template, input_variables=['question'])

# 创建调用chain
llm_chain = LLMChain(prompt = prompt, llm = llm_flan_t5, verbose = True)

# FLAN-T5不支持中文
input_text = 'Who is current president of USA?'
print(input_text)
print(llm_chain.run(input_text))
```

你可以把一系列 LLM 和 prompt提问 组合成调用链 Chain，来完成特定任务，Chain就是完成某个具体任务的基本单元。多个 Chain 可以再组装成更大的 Chain，完成更加复杂的任务。

```py
llm = OpenAI(temperature = 0.9)
prompt = PromptTemplate(input_variables = ['country'], template = '请为喜欢{country}的人推荐5款汽车型号')
llm_chain = LLMChain(llm = llm, prompt = prompt)

while True:
   print('\n请输入某个国家，LLM将为喜欢这个国家的人推荐5款汽车')
   input_text = input()
   print('Current PromptTemplate is :%s'%(prompt.format(country = input_text)))
   print(llm_chain.run(input_text))  
```

Agent 是 LangChain 中的一个重要概念，它是一系列流程的封装，动态串联多个 Tool 或 Chain，完成对复杂问题的自动推导和执行解决过程。

Tool可以认为是Agent中每个单独功能的封装，这些功能可以由第三方在线服务，本地服务，本地可执行程序等不同方式实现。用户写好 Tool 的描述和调用外部插件的接口之后，LLM会自动决定在什么场景下使用，非常智能。

```py
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# SerpAPI提供的API Key，即开发者的Secret Key
os.environ['SERPAPI_API_KEY'] = config['keys']['serpapi_api_key']

# 创建OpenAI的默认LLM
llm = OpenAI(temperature = 0.9)

# 加载搜索引擎tool和预定义的llm-math chain，llm-math会使用python命令行处理数学计算问题，并返回结果
tools = load_tools(['serpapi', 'llm-math'], llm = llm)

# Initialize agent with 3 params:
# 1. a service tool
# 2. a LLM
# 3. agent type
agent = initialize_agent(tools, llm, agent = 'zero-shot-react-description', verbose = True)

while True:
    print('请输入一个包含数学计算的问题，LLM将结合Google搜索引擎对问题进行拆解并使用数学工具进行计算，过程是通过Agent自动实现的')
    input_text = input()
    print(agent.run(input_text)) 
```


memory模块：langchain.memory: 原始的Chain是无状态的，就是与之对话，没有记忆功能。因此引入Memory模块，来赋能agent具有上下文记忆的能力。

```py
llm = OpenAI(temperature = 0)
conversation = ConversationChain(llm = llm, verbose = True)

print('开始一个新的对话，LLM是OpenAI的text-davinci-003')

while True:
    input_text = input()
    conversation.predict(input = input_text) 
    
# 在ConversationChain中使用的记忆提示模板
# _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
# Current conversation:
# {history}
# Human: {input}
# AI:"""
```


langchain.indexs: Index是用于将文档结构化的方法，非结构化文档结构化后才可以更好地和LLM进行交互。最常用的index类型是向量类型，例如文本的embedding结果，因此Index包含了对向量处理的相关功能函数。


LangChain 的优势：
* 基于langchain编程库，可以通过chain创建基于人工设计的各项能力组件（llm，prompt，memory，index，子chain等）的执行流程；通过Tool封装扩展能力或服务（如搜索引擎API, 数学计算API等）；通过Agent创建自动判断、动态编排的各项tool的执行流程；
* 同时，langchain对prompt模板、对话记忆、向量存储计算提供了基础工具函数。在langchain中，LLM的主要作用是对语言进行理解和问答生成。langchain支持用户自定义tool，chain，agent和prompt等，具有很强的可扩展性。

LangChain 的局限：
* Agent的智能化水平依赖于LLM的请求理解和任务拆分能力，有时会出现对问题拆解的错误，导致错误的执行流程和最终错误的结果。
* 由于过程中Agent可能存在多次与LLM的交互，整体的响应延时在分钟至十分钟级，不适合需要秒级以下响应的场景。


#### 4、插件

LangChain 是构建第三方应用的比较方便的框架。但如果你只是想为LLM添加某个能力，则可以直接注册插件，并在官网上调用即可。这一点 Chatgpt 和 文心一言都可以做到，并且开发过程完全一致。

ChatGPT Plugins是由OpenAI建立的一套面向开发者的插件协议和规范。这些插件在OpenAI注册并审核通过后，可以在用户和ChatGPT产品对话的过程中，由ChatGPT背后的LLM自行判断调用，从而实现对ChatGPT及其LLM能力的扩展。

插件的运行原理是，用户输入query语句（如“生成汽车信息的广告脚本”）后，
1. LLM 根据设定的词槽（如“业务点”、“品牌”、“行业”等），提取query中的对应的信息，这个提取过程也是LLM理解的；即：将自然语言文本转化为结构化的信息；
2. 后端根据结构化信息，查询指定的服务，并返回相应的信息；
3. LLM根据提取的信息和历史问答，生成下一步的回答。

用户选择插件后，LLM会判断用户的query是否满足插件唤醒的要求；如果插件包含多个功能，LLM还会判断该query具体要执行哪个功能；最后LLM会根据功能对应的词槽，提取对应的信息。

由此，如果你有一个能力接口，你想把它用在LLM上，则只需描述你需要的信息，LLM就能在合适的场景自动调用你的能力。

#### 参考文档

https://www.langchain.asia/getting_started/getting_started
https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide
https://liaokong.gitbook.io/llm-kai-fa-jiao-cheng/