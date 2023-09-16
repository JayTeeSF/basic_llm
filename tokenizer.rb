class Tokenizer
  # should this be a long string or an array
  # theoretically it could be a file_name...
  def self.tokenize(str)
      str.split
  end

  def self.from_string(str)
    new(tokenize(str))
  end

  def self.from_file(file_path)
    new(tokenize(File.read(file_path)))
  end

  attr_reader :vocab 
  # array of ids if from embeddings.vocab!
  # (from tokenizer.encodings)
  # But I'd like to have words...
  def initialize(inputs)
    @vocab = inputs
    @word_to_token = {}
    @token_to_word = {}
    
    @vocab.each_with_index do |word, idx|
      # warn("vocab #{word.inspect} => #{idx}")
      @word_to_token[word] = idx
      @token_to_word[idx] = word
    end
  end

  def encodings
    encode(vocab)
  end

  def tokenize(str)
    self.class.tokenize(str)
  end

  def encode(token_str_ary)
    token_str_ary.map {|token_str| encode_token(token_str) }
  end

  def decode(token_id_ary)
    token_id_ary.map {|token| decode_token(token) }
  end

  private

  def encode_token(word)
    @word_to_token[word]
  end

  def decode_token(token)
    @token_to_word[token]
  end
end
