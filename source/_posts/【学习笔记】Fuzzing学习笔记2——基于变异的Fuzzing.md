---
title: 【学习笔记】Fuzzing学习笔记2——基于变异的Fuzzing
date: 2020-08-19 23:48:14
math: false
index_img: /img/softwaretesting.jpg
tags: ['Fuzzing', 'Testing']
categories: 
- notes
---
基于变异的模糊测试。
<!--more--->

https://www.fuzzingbook.org/html/MutationFuzzer.html

大多数随机生成的输入在语法上都是无效的，程序很快就会检测并拒绝这些输入，这样便达不到深入测试Runner内部的目的。因此我们必须试图生成有效的输入。

本节我们介绍Mutation Fuzzing，这种基于变异的方法对现有输入进行小的更改，这些更改可能仍使输入保持有效，但仍会表现出新的行为。 

要对字符串形式的输入进行变异（Mutate），具体来说，就是执行随机插入字符、删除字符、修改字符等操作。Mutational fuzzing的特点是基于一个有效的原始输入，与之前凭空捏造出来一个input的generational fuzzing不同。

随机删除

```py
def delete_random_character(s):
    """Returns s with a random character deleted"""
    if s == "":
        return s

    pos = random.randint(0, len(s) - 1)
    # print("Deleting", repr(s[pos]), "at", pos)
    return s[:pos] + s[pos + 1:]

seed_input = "A quick brown fox"
for i in range(10):
    x = delete_random_character(seed_input)
    print(repr(x))

'A uick brown fox'
'A quic brown fox'
'A quick brown fo'
'A quic brown fox'
'A quick bown fox'
'A quick bown fox'
'A quick brown fx'
'A quick brown ox'
'A quick brow fox'
'A quic brown fox'
```

随机插入

```py
def insert_random_character(s):
    """Returns s with a random character inserted"""
    pos = random.randint(0, len(s))
    random_character = chr(random.randrange(32, 127))
    # print("Inserting", repr(random_character), "at", pos)
    return s[:pos] + random_character + s[pos:]

for i in range(10):
    print(repr(insert_random_character(seed_input)))

'A quick brvown fox'
'A quwick brown fox'
'A qBuick brown fox'
'A quick broSwn fox'
'A quick brown fvox'
'A quick brown 3fox'
'A quick brNown fox'
'A quick brow4n fox'
'A quick brown fox8'
'A equick brown fox'
```

随机替换

```py
def flip_random_character(s):
    """Returns s with a random bit flipped in a random position"""
    if s == "":
        return s

    pos = random.randint(0, len(s) - 1)
    c = s[pos]
    bit = 1 << random.randint(0, 6)
    new_c = chr(ord(c) ^ bit)
    # print("Flipping", bit, "in", repr(c) + ", giving", repr(new_c))
    return s[:pos] + new_c + s[pos + 1:]

for i in range(10):
    print(repr(flip_random_character(seed_input)))

'A quick bRown fox'
'A quici brown fox'
'A"quick brown fox'
'A quick brown$fox'
'A quick bpown fox'
'A quick brown!fox'
'A 1uick brown fox'
'@ quick brown fox'
'A quic+ brown fox'
'A quick bsown fox'
```

只要我们有一些原始输入，这些输入是有效的，那么我们基于原始输入的变异也应该是有效的。

多重变异

假设我们这里有个方法mutate()，能对字符串执行变异操作。那么连续变异50次，输入会变成什么样子？
```py
seed_input = "http://www.google.com/search?q=fuzzing"
mutations = 50

inp = seed_input
for i in range(mutations):
    if i % 5 == 0:
        print(i, "mutations:", repr(inp))
    inp = mutate(inp)

0 mutations: 'http://www.google.com/search?q=fuzzing'
5 mutations: 'http:/L/www.googlej.com/seaRchq=fuz:ing'
10 mutations: 'http:/L/www.ggoWglej.com/seaRchqfu:in'
15 mutations: 'http:/L/wwggoWglej.com/seaR3hqf,u:in'
20 mutations: 'htt://wwggoVgle"j.som/seaR3hqf,u:in'
25 mutations: 'htt://fwggoVgle"j.som/eaRd3hqf,u^:in'
30 mutations: 'htv://>fwggoVgle"j.qom/ea0Rd3hqf,u^:i'
35 mutations: 'htv://>fwggozVle"Bj.qom/eapRd[3hqf,u^:i'
40 mutations: 'htv://>fwgeo6zTle"Bj.\'qom/eapRd[3hqf,tu^:i'
45 mutations: 'htv://>fwgeo]6zTle"BjM.\'qom/eaR[3hqf,tu^:i'
```

可以看到变异体已经几乎无法识别了。我们通过多次变异，获得了更加多样的输入。

MutationFuzzer的实现

```py
class MutationFuzzer(Fuzzer):
    def __init__(self, seed, min_mutations=2, max_mutations=10):
        self.seed = seed
        self.min_mutations = min_mutations
        self.max_mutations = max_mutations
        self.reset()

    def reset(self):
        self.population = self.seed
        self.seed_index = 0

    def mutate(self, inp):
        return mutate(inp)

    def create_candidate(self):
        candidate = random.choice(self.population)
        trials = random.randint(self.min_mutations, self.max_mutations)
        for i in range(trials):
            candidate = self.mutate(candidate)
        return candidate

    def fuzz(self):
        if self.seed_index < len(self.seed):
            # Still seeding
            self.inp = self.seed[self.seed_index]
            self.seed_index += 1
        else:
            # Mutating
            self.inp = self.create_candidate()
        return self.inp
```

`create_candidate()`随机选取种子`candidate`，然后将这个种子随机突变`trials`次，返回经过多次突变的`candidate`。

` fuzz()`方法一开始返回的是未经突变的种子样本，当种子挑选完毕后，返回突变样本。这样可以确保每次调用fuzz()，得到的输出是不一样的。

Mutational Fuzzing成功的关键在于引导这些突变的方法--即保留那些特别有价值的样本。

覆盖率引导

我们可以利用被测程序来引导测试用例生成。以前我们只是收集程序执行成功或者失败的信息，现在我们可以收集多点信息，比如运行时代码覆盖率。

利用覆盖率引导变异的Fuzzing，最成功的实践是[American fuzzy loop](http://lcamtuf.coredump.cx/afl/)，即AFL。

AFL会生成“成功”的测试用例。AFL认为，所谓“成功”是指找到了一条新的程序执行路径。AFL不断地突变新路径的输入，如果产生了新的路径，输入会保留下来。

为了获得程序运行时的覆盖率信息，我们需要重新定义Runner。FunctionRunner类负责包装一个被测函数。

```py
class FunctionRunner(Runner):
    def __init__(self, function):
        """Initialize.  `function` is a function to be executed"""
        self.function = function

    def run_function(self, inp):
        return self.function(inp)

    def run(self, inp):
        try:
            result = self.run_function(inp)
            outcome = self.PASS
        except Exception:
            result = None
            outcome = self.FAIL

        return result, outcome
```

而FunctionCoverageRunner在此基础上增加了覆盖率计算模块`Coverage`。

```py
class FunctionCoverageRunner(FunctionRunner):
    def run_function(self, inp):
        with Coverage() as cov:
            try:
                result = super().run_function(inp)
            except Exception as exc:
                self._coverage = cov.coverage()
                raise exc

        self._coverage = cov.coverage()
        return result

    def coverage(self):
        return self._coverage
```

下面改写Fuzzer类。

```py
class MutationCoverageFuzzer(MutationFuzzer):
    def reset(self):
        super().reset()
        self.coverages_seen = set()
        # Now empty; we fill this with seed in the first fuzz runs
        self.population = []

    def run(self, runner):
        """Run function(inp) while tracking coverage.
           If we reach new coverage,
           add inp to population and its coverage to population_coverage
        """
        result, outcome = super().run(runner)
        new_coverage = frozenset(runner.coverage())
        if outcome == Runner.PASS and new_coverage not in self.coverages_seen:
            # We have new coverage
            self.population.append(self.inp)
            self.coverages_seen.add(new_coverage)

        return result
```
`MutationCoverageFuzzer`内部保存测试用例队列`population`和覆盖率队列`coverages_seen`。如果fuzz的input产生了新的coverage，则将该input添加到population中，并将该coverage添加到coverage_seen中。

由此，我们得到的population中的每个input都能够使得程序产生不同的coverage，这背后可能是程序的不同执行路径，也就增加了inputs的多样性。
