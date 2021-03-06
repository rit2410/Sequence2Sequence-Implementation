# Sequence2Sequence-Implementation

The Seq2Seq framework relies on the encoder-decoder paradigm. The encoder encodes the input sequence, while the decoder produces the target sequence

### Encoder

Our input sequence is how are you. Each word from the input sequence is associated to a vector  π€ββπ  (via a lookup table). In our case, we have 3 words, thus our input will be transformed into  [π€0,π€1,π€2]ββπΓ3 . Then, we simply run an LSTM over this sequence of vectors and store the last hidden state outputed by the LSTM: this will be our encoder representation  π . Letβs write the hidden states  [π0,π1,π2]  (and thus  π=π2 )

## Decoder

Now that we have a vector  π  that captures the meaning of the input sequence, weβll use it to generate the target sequence word by word. Feed to another LSTM cell:  π  as hidden state and a special start of sentence vector  π€π ππ   as input. The LSTM computes the next hidden state  β0βββ . Then, we apply some function  π:βββ¦βπ  so that  π 0:=π(β0)ββπ  is a vector of the same size as the vocabulary.

β0π 0π0π0=LSTM(π,π€π ππ )=π(β0)=softmax(π 0)=argmax(π0)
 
Then, apply a softmax to  π 0  to normalize it into a vector of probabilities  π0ββπ  . Now, each entry of  π0  will measure how likely is each word in the vocabulary. Letβs say that the word βcommentβ has the highest probability (and thus  π0=argmax(π0)  corresponds to the index of βcommentβ). Get a corresponding vector  π€π0=π€πππππππ‘  and repeat the procedure: the LSTM will take  β0  as hidden state and  π€πππππππ‘  as input and will output a probability vector  π1  over the second word, etc.

β1π 1π1π1=LSTM(β0,π€π0)=π(β1)=softmax(π 1)=argmax(π1)
 
The decoding stops when the predicted word is a special end of sentence token.
