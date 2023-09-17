#!/usr/bin/env ruby

require_relative "./full.rb"

full_cmd = './full.rb 2>&1'
full_out = %x(#{full_cmd})
grep_cmd = "grep 'transduction' corpus.txt"
grep_out = %x(#{grep_cmd})
cat_cmd = "cat full.rb"
cat_out = %x(#{cat_cmd})

enum_instructions = <<-EOF
The goal is to build an attention-based language model like chatGPT or Claude, where:
  -  Input Embedding Layer: Convert input words/tokens into numerical vector representations. This allows the model to understand semantic relationships between words.
  -  Self-Attention Layers: Relate different words in a sentence to each other, helping the model understand context. The core mechanism is computing compatibility scores between words and using those to aggregate information.
  -  Feedforward Layers: Process the intermediate representations from the attention layers to extract higher-level features.
  -  Output Layer: Convert the features from the last feedforward layer into predicted next-word probabilities.

  The key is stacking multiple self-attention and feedforward layers between the in put and output, so that we have a full model that can learn complex relationships between words and sentence structure.

Therefore, given our progress to-date, your remaining instructions are:
1. Please find and fix the bugs in the full.rb code (see below). 
2. If necessary, update the script that calls these classes to use an example that is supported by the sample from corpus.txt (see below).
3. Use an unknown (UNK) token and supporting-code to handle unknown characters
4. Complete any missing methods or method-logic.
5. If anything is missing from the basic design (see below), 
add it (i.e. missing classes or logic).
6. If the basic design is complete say, "the design is complete".
7. Per the basic design, update the script that runs the code to indicate the output at the barrier of each module (i.e. the tokens each arrow represents passing).
EOF

basic_design = <<-EOF
Here is the basic design:
```
Input Words         ->   Input Embedding Layer   -> Embedded Word Vectors

Embedded Vectors    ->   Self-Attention Layer    -> Contextual Vectors  

Context Vectors     ->   Feedforward Layer      -> Higher-Level Features

Higher-Level Features -> Output Layer       -> Next Word Probabilities
```
EOF

code_run = <<-EOF
Here is the output from running the code:
```
#{full_out}
```
EOF

corpus = <<-EOF
Here is a sample of my corpus.txt
`#{grep_cmd}`
```
#{grep_out}
```
EOF


codez = <<-EOF
Below is the current code:
```
#{cat_out}
```
EOF

# fail("trying to generate code_run: #{code_run}")

puts <<-EOF
#{enum_instructions}

#{basic_design}

#{code_run}

#{corpus}

#{codez}
EOF
