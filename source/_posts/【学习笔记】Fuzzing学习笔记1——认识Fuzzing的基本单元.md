---
title: 【学习笔记】Fuzzing学习笔记1——认识Fuzzing的基本单元
date: 2020-08-19 20:10:10
math: false
index_img: /img/softwaretesting.jpg
tags: ['Fuzzing', 'Testing']
categories: 
- notes
---
Fuzzing又称模糊测试。
<!--more--->

https://www.fuzzingbook.org/html/Fuzzer.html

## Fuzzing测试是什么？

> Create random inputs, and see if they break things.

说得简单、纯粹点，Fuzzing是一种软件测试方法，通过不断生成不同的输入，使被测程序崩溃、出错，以此改进程序本身的软件测试方法。由于Fuzzing的核心在于生成软件测试用例，因此这种方法又被称为**生成软件测试**。

## 构建第一个fuzzer
```py
import random

def fuzzer(max_length=100, char_start=32, char_range=32):
    """A string of up to `max_length` characters
       in the range [`char_start`, `char_start` + `char_range`]"""
    string_length = random.randrange(0, max_length + 1)
    out = ""
    for i in range(0, string_length):
        out += chr(random.randrange(char_start, char_start + char_range))
    return out

fuzzer()
4$)>(,-&!$25;>6=27= 5)9?300(.466&('$*,,1:8' ,$/99>'*=(
```

此fuzzer的作用是生成一堆随机字符。要想只生成26个字母，那么

```py
fuzzer(100, ord('a'), 26)
ueffzgwltwmspvmowihhtjmgsixofnvntnqmr
```

## Fuzzing关键的两个部件：

### Fuzzer类

`Fuzzer`，是所有`fuzzer`的基类，`RandomFuzzer` 是其简单实现。`Fuzzer` 的`fuzz()`接口返回一个字符串，字符串内容是根据不同实现逻辑而构造出来的。

比如Fuzzer的实现RandomFuzzer，其`fuzz()`就是随机生成的字符串。

```py
>>> random_fuzzer = RandomFuzzer(min_length=10, max_length=20, char_start=65, char_range=26)
>>> random_fuzzer.fuzz()
'XGZVDDPZOOW'
```
Fuzzer的run()接口负责运行一个Runner对象。

下面是Fuzzer的代码架构：

```py
class Fuzzer(object):
    def __init__(self):
        pass

    def fuzz(self):
        """Return fuzz input"""
        return ""

    def run(self, runner=Runner()):
        """Run `runner` with fuzz input"""
        return runner.run(self.fuzz())

    def runs(self, runner=PrintRunner(), trials=10):
        """Run `runner` with fuzz input, `trials` times"""
        # Note: the list comprehension below does not invoke self.run() for subclasses
        # return [self.run(runner) for i in range(trials)]
        outcomes = []
        for i in range(trials):
            outcomes.append(self.run(runner))
        return outcomes
```

此时Fuzzer基类的fuzz()接口还没有功能。派生类RandomFuzzer则实现了fuzz()：

```py
class RandomFuzzer(Fuzzer):
    def __init__(self, min_length=10, max_length=100,
                 char_start=32, char_range=32):
        """Produce strings of `min_length` to `max_length` characters
           in the range [`char_start`, `char_start` + `char_range`]"""
        self.min_length = min_length
        self.max_length = max_length
        self.char_start = char_start
        self.char_range = char_range

    def fuzz(self):
        string_length = random.randrange(self.min_length, self.max_length + 1)
        out = ""
        for i in range(0, string_length):
            out += chr(random.randrange(self.char_start,
                                        self.char_start + self.char_range))
        return out
```

有了RandomFuzzer，我们可以生成一些随机的字符串了。

```py
random_fuzzer = RandomFuzzer(min_length=20, max_length=20)
for i in range(10):
    print(random_fuzzer.fuzz())
'>23>33)(&"09.377.*3
*+:5 ? (?1$4<>!?3>.'
4+3/(3 (0%!>!(+9%,#$
/51$2964>;)2417<9"2&
907.. !7:&--"=$7',7*
(5=5'.!*+&>")6%9)=,/
?:&5) ";.0!=6>3+>)=,
6&,?:!#2))- ?:)=63'-
,)9#839%)?&(0<6("*;)
4?!(49+8=-'&499%?< '
```

### Runner类

`Runner`，是所有待测程序的基类。一个Fuzzer 与一个Runner搭配。

Runner类含有run(input)接口，负责接收input并执行，返回 (result, outcome)，result是Runner在运行时的信息和细节，而outcom代表着这次运行的结果。

运行结果为枚举对象 outcome，含义为程序运行结果，有(PASS, FAIL, or UNRESOLVED)三种可能。

* Runner.PASS：测试通过，run()输出正确。
* Runner.FAIL：测试失败，结果错误。
* Runner.UNRESOLVED：没有输出，这一般代表runner无法应对输入而崩溃。

Runner的大体架构如下：

```py
class Runner(object):
    # Test outcomes
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    def __init__(self):
        """Initialize"""
        pass

    def run(self, inp):
        """Run the runner with the given input"""
        return (inp, Runner.UNRESOLVED)
```

想要实现其他Runner，只需继承Runner即可。

```py
class PrintRunner(Runner):
    def run(self, inp):
        """Print the given input"""
        print(inp)
        return (inp, Runner.UNRESOLVED)
```

```
p = PrintRunner()
(result, outcome) = p.run("Some input")
Some input
```

对于PrintRunner，我们无法验证其结果，因此通通返回UNRESOLVED。

下面是一个Runner的派生类ProgramRunner的代码，此架构代表了大多数程序。

```py
class ProgramRunner(Runner):
    def __init__(self, program):
        """Initialize.  `program` is a program spec as passed to `subprocess.run()`"""
        self.program = program

    def run_process(self, inp=""):
        """Run the program with `inp` as input.  Return result of `subprocess.run()`."""
        return subprocess.run(self.program,
                              input=inp,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True)

    def run(self, inp=""):
        """Run the program with `inp` as input.  Return test outcome based on result of `subprocess.run()`."""
        result = self.run_process(inp)

        if result.returncode == 0:
            outcome = self.PASS
        elif result.returncode < 0:
            outcome = self.FAIL
        else:
            outcome = self.UNRESOLVED

        return (result, outcome)
```

## Fuzzing 实例

```py
cat = ProgramRunner(program="cat")
cat.run("hello")

random_fuzzer = RandomFuzzer(min_length=20, max_length=20)

random_fuzzer.runs(cat, 10)

[(CompletedProcess(args='cat', returncode=0, stdout='3976%%&+%6=(1)3&3:<9', stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout='33$#42$ 11=*%$20=<.-', stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout='"?<\'#8 </:*%9.--\'97!', stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout="/0-#(03/!#60'+6>&&72", stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout="=,+:,6'5:950+><3(*()", stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout=" 379+0?'%3137=2:4605", stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout="02>!$</'*81.#</22>+:", stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout="=-<'3-#88*%&*9< +1&&", stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout='2;;0=3&6=8&30&<-;?*;', stderr=''),
  'PASS'),
 (CompletedProcess(args='cat', returncode=0, stdout='/#05=*3($>::#7!0=12+', stderr=''),
  'PASS')]
```
