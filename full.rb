#!/usr/bin/env ruby

require 'matrix'

class Token
  attr_reader :id, :text

  def initialize(id, text)
    @id = id
    @text = text
  end

  def to_i
    id
  end

  def to_s
    "#{@id}) #{@text}" 
  end
  UNK = new(0, '<UNK>')
end

class Embeddings
  def self.from_file(corpus_file)
    # Load corpus
    corpus = File.read(corpus_file)

    # Extract vocabulary
    new(corpus.split)
  end

  def initialize(word_token_array)
    @embeddings = {}

    freqs = Hash.new(0)
    word_token_array.each_with_index do |wt, i|
      freqs[Token.new(i, wt)] += 1
    end

    vocab = freqs.keys.sort_by{ |t| freqs[t] }.reverse[0..10]

    vocab.each do |token|
      @embeddings[token] = Array.new(10) { rand }
    end
  end

  def size
    @embeddings.keys.size
  end

  def lookup_by_id(token_id)
    @embeddings.find {|t| t.id == token_id}
  end

  def lookup(token)
    @embeddings[token]
  end

  def [](token)
    lookup(token)
  end

  def include?(token)
    !! lookup(token)
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
    z1 = input * @w1 + @b1
    a1 = relu(z1)
    z2 = a1 * @w2 + @b2
    a2 = z2
    return a2
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
    return probs
  end

  private

  def softmax(scores)
    total = scores.values.sum
    probs = scores.each { |k, v| scores[k] = v / total }
    probs
  end
end

class Attention
  def initialize(embeddings)
    @embeddings = embeddings
    @feedforward = Feedforward.new(embeddings.size, 20)
    @output_layer = OutputLayer.new(20, embeddings.size)
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
    output = generate_output(vectors)

    return output
  end

  private

  def embed(input_tokens)
    # Look up the embeddings for the input tokens
    input_embeddings = input_tokens.map do |token|
      @embeddings[token]
      #unless got
      #  @embeddings[Token::UNK] =
      #end
      #@embeddings.include?(token) ? @embeddings[token] : @embeddings[Token::UNK]
      # [token] || @embeddings[Token::UNK]
    end

    print "Input Embedding Layer: \n#{input_embeddings.map(&:to_s)}\n\n"

    # Return the input embeddings
    return input_embeddings.map(&:id)
  end

  # Self-attention(input_embeddings)
  def self_attend(vectors)
    # Compute compatibility scores
    scores = {}
    vectors.each_with_index do |query, i|
      vectors.each_with_index do |key, j|
        scores[[i, j]] = dot_product(query, key)
      end
    end

    # Apply softmax
    weights = softmax(scores)

    # Calculate weighted average
    attended = []
    # vectors.each_with_index do |_value, i|
    vectors.size.times do |i|
      weighted_sum = 0

      weights.each do |(i2, j), weight|
        if i == i2
          weighted_sum += weight * vectors[j] 
        end
      end
      attended << weighted_sum
    end

    print "Self-Attention Layer: \n#{attended}\n\n"

    return attended
  end

  # dot_product(query, key)
  def dot_product(x, y)
    # query.transpose * key
    x.inner_product(y)
  end

  def generate_output(vectors)
    output = @output_layer.forward(vectors)
    print "Output:\n" if debug?
    pp output if debug?
    output
  end

  def feedforward(vectors)
    outputs = @feedforward.forward(vectors)
    print "Feedforward outputs:\n" if debug?
    pp outputs if debug?
    outputs
  end
end

class Tokenizer
  def initialize(embeddings)
    @embeddings = embeddings
  end

  def tokenize(text)
    tokens = text.split(' ')

    # Convert to Tokens
    tokenized = []
    tokens.each_with_index do |token, i|  
      if @embeddings.include?(token)
        tokenized << Token.new(i, token)
      else
        tokenized << Token::UNK
      end
    end

    tokenized
  end
end

if __FILE__ == $PROGRAM_NAME
  corpus_file = 'corpus.txt'
  embeddings = Embeddings.from_file(corpus_file)
  tokenizer = Tokenizer.new(embeddings)

  input = "the dominant sequence transduction models"
  tokens = tokenizer.tokenize(input)

  attention = Attention.new(embeddings)
  pp tokens if attention.debug?

  raw_output = attention.attend(tokens)
  output = raw_output.map {|t_id| embeddings.lookup_by_id(t_id) }

  puts "Output: #{output}"
end
