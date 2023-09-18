#!/usr/bin/env ruby

require 'matrix'

class Token
  attr_reader :id, :word

  def initialize(id, word)
    @id = id
    @word = word
  end

  def to_i
    id
  end

  def to_s
    "#{id}: >>#{@word}<<" 
  end
  UNK = new(0, '<UNK>')
end

class Embeddings
  def self.from_file(corpus_file)
    # Load corpus
    corpus = File.read(corpus_file)

    new(corpus)
  end

  attr_reader :vocab, :vocab_size
  attr_reader :word_to_token, :id_to_token # DEBUG ONLY

  def initialize(text, vocab_size: 20_000, size: 10)
    @vocab_size = vocab_size
    @size = size
    word_token_array = Tokenizer.split(text)
    @embeddings = {}

    freqs = Hash.new(0)
    word_token_array.each_with_index do |wt, i|
      freqs[Token.new(i, wt)] += 1
    end

    # Extract vocabulary, from most frequent tokens
    @vocab = freqs.keys.sort_by{ |t| freqs[t] }.reverse[0..@vocab_size]
    @word_to_token = @vocab.reduce({}) {|m,t| m[t.word] = t; m }
    @id_to_token = @vocab.reduce({}) {|m,t| m[t.id] = t; m }

    @vocab.each do |token|
      # @embeddings[token] = Array.new(@size) { rand }
      @embeddings[token] = Matrix.rows([[rand]] * size)
      # @embeddings[token] = Matrix.column_vector([[rand]] * size)
    end
  end

  def size
    @embeddings.keys.size
  end

  def lookup_by_id(token_id)
    @id_to_token[token_id]
  end

  def lookup(token)
    # @embeddings[token]
    @embeddings[token] #.to_a
  end

  def [](token)
    lookup(token)
  end

  def find_by_word(wt)
    @word_to_token[wt]
  end
end

class Feedforward
  def initialize(embedding_dim, hidden_dim)
    @w1 = Matrix.rows([[ Random.rand ] * embedding_dim])
    @b1 = Matrix.zero(hidden_dim, 1)
    @w2 = Matrix.rows([[ Random.rand ] * hidden_dim])
    @b2 = Matrix.zero(embedding_dim, 1)
  end

  def forward(input)
    # input = Matrix.rows([[1, 2, 3], [4, 5, 6]])
    # w1 = Matrix.rows([[7, 8, 9], [10, 11, 12]])
    # b1 = Matrix.rows([[13, 14, 15]])

    # Convert the input Array to a Matrix
    input_matrix = Matrix.rows([input])
    input_vector = input_matrix.to_vector

    z1 = input_vector * w1 + b1
    # z1 = input * @w1 + @b1
    a1 = relu(z1)
    z2 = a1 * @w2 + @b2
    a2 = z2
    a2
  end

  private

  def relu(x)
    x.map { |v| v.positive? ? v : 0 }
  end
end

class OutputLayer
  def initialize(embedding_dim, vocab_size)
    @w = Matrix.rows([[ Random.rand ] * embedding_dim])
    @b = Matrix.zero(vocab_size, 1)
  end

  def forward(input)
    logits = input * @w + @b
    probs = softmax(logits)
    probs
  end

  def softmax(scores)
    #total = scores.values.sum
    #probs = scores.each { |k, v| scores[k] = v / total }
    #probs
    exp_scores = scores.map { |k, v| [k, Math.exp(v)] }
    total = exp_scores.map { |_, v| v }.sum
    norm_scores = exp_scores.map { |k, v| [k, v / total] }
    norm_scores.to_h
  end
end

class Attention
  def initialize(embeddings)
    @embeddings = embeddings
    @feedforward = Feedforward.new(embeddings.size, embeddings.size)
    @output_layer = OutputLayer.new(embeddings.size, embeddings.size)
  end

  def debug?
    true # Change to enable debug prints
  end

  def attend(input_tokens)
    # Embed input tokens
    input_embeddings = embed(input_tokens)

    # Stack self-attention and feedforward layers
    num_layers = 6
    vectors = input_embeddings
    # for i in 1..num_layers
    num_layers.times do
      # Self-attention 
      vectors = self_attend(vectors)

      # Feedforward
      vectors = feedforward(vectors)
    end

    # Output layer
    generate_output(vectors)
  end

  private

  def embed(input_tokens)
    # Look up the embeddings for the input tokens
    input_embeddings = input_tokens.map do |token|
      @embeddings[token]
    end

    # print "Input Embedding Layer: \n#{input_embeddings.map { |v| v.map(&:to_s) }}\n\n"
    # print "1) Input Embedding Layer: \n#{input_embeddings.map(&:to_s)}\n\n"
    print "Input Embedding Layer: #{input_embeddings}\n\n"

    # Return the input embeddings
    input_embeddings #.map { |v| v.map(&:to_i) }
  end

  # Self-attention(input_embeddings)
  def self_attend(vectors)
    # Map input vectors to arrays
    # arrays = vectors.map(&:to_a)
    # columns = vectors.map(&:column)
    columns = vectors.map { |v| v.column(0) }

    # Convert input to matrix
    # vectors = Matrix.rows(arrays)
    # vectors = Matrix.columns(arrays)
    vectors = Matrix.rows(columns)

    # Compute compatibility scores
    scores = {}
    vectors.row_vectors.each_with_index do |query, i|
      vectors.row_vectors.each_with_index do |key, j|  
        scores[[i, j]] = query.dot(key)
      end 
    end

    # Apply softmax
    weights = @output_layer.softmax(scores)

    # Calculate weighted average
    attended = []

    # vectors.size.times do |i|
    vectors.row_count.times do |i|
      weighted_sum = 0

      weights.each do |(i2, j), weight|
        weighted_sum += weight * vectors[i, j] if i == i2
        #  vectors[j]

      end
      attended << weighted_sum
    end

    print "Self-Attention Layer: \n#{attended}\n\n"

    attended
  end

  def dot_product(query, key)
    # query.transpose * key
    # query.inner_product(key)
    query.dot(key) 
  end

  def generate_output(vectors)
    output = @output_layer.forward(vectors)
    print "Output Layer: \n#{output}\n\n" if debug?
    output
  end

  def feedforward(vectors)
    outputs = @feedforward.forward(vectors)
    print "Feedforward outputs:\n #{outputs}\n\n" if debug?
    outputs
  end
end

class Tokenizer
  def initialize(embeddings)
    @embeddings = embeddings
  end

  def self.split(text)
    text.split(' ')
  end
  def tokenize(text)
    word_tokens = Tokenizer.split(text)

    # Convert to Word Tokens
    tokenized = []
    # word_tokens.each_with_index do |wt, i| # does it make sense to add an index to the tokens ...it won't match the embeddings?!
    word_tokens.each do |wt|
      token = @embeddings.find_by_word(wt) || Token::UNK
      warn("Adding #{token} for #{wt}...")
      tokenized << token
    end

    tokenized
  end
end

if __FILE__ == $PROGRAM_NAME
  corpus_file = 'corpus.txt'
  embeddings = Embeddings.from_file(corpus_file)
  raise "All embedding tokens are UNK" if embeddings.vocab.all? {|t| t == Token::UNK }

  tokenizer = Tokenizer.new(embeddings)

  input = "transduction problems such as language"
  tokens = tokenizer.tokenize(input)

  attention = Attention.new(embeddings)

  if attention.debug?
    warn "Actual vocab size is: #{embeddings.vocab.size}, and they're not all <UNK> tokens, or this program would have already raised an exception"
    pp "tokens from input: >>#{tokens}<<"
    # puts 'Stringified Embedding Tokens:'
    # embeddings.word_to_token.each {|w| puts("\t#{w}") }

    embedded = attention.send(:embed,tokens)
    print "tmp) Input Embedding Layer: \n#{embedded}\n\n"

    attended = attention.send(:self_attend,embedded)
    print "Self-Attention Layer: \n#{attended}\n\n"

    fedforward = attention.send(:feedforward,attended)
    print "Feedforward Layer: \n#{fedforward}\n\n"

    output = attention.send(:generate_output,fedforward)
    print "Output Layer: \n#{output}\n\n"
  end

  raw_output = attention.attend(tokens)
  # output = raw_output.map {|t_id| embeddings.lookup_by_id(t_id) }
  # output = raw_output.map {|vec| vec }
  output = raw_output.map {|t| t.to_s }

  puts "Output: #{output}"
end
