#https://github.com/minimaxir/textgenrnn/blob/master/setup.py
#https://en.wikipedia.org/wiki/Recurrent_neural_network

#install_requires=['keras>=2.1.5', 'h5py', 'scikit-learn', 'tqdm']

from textgenrnn import textgenrnn



textgen = textgenrnn(weights_path='gibberish_weights.hdf5',
                       vocab_path='gibberish_vocab.json',
                       config_path='gibberish_config.json')


textgen.generate_samples(max_gen_length=1000)
textgen.generate_to_file('gibberishoutput.txt', max_gen_length=1000)