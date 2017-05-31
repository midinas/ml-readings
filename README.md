## ml-readings
List of machine learning resources I found most valuable 
 
### Graphical Models
- [Introduction to Markov Random Fields](http://www.cs.toronto.edu/~kyros/courses/2503/Handouts/Blake2011.pdf)
- [Graphical Models, Exponential Families, and Variational Inference](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf)
- [Classic HMM Intro Paper Rabiner](http://www.cs.umb.edu/~rvetro/vetroBioComp/HMM/Rabiner1986%20An%20Introduction%20to%20Hidden%20Markov%20Models.pdf)

#### Probabilistic Modeling
- [Edward python lib](http://edwardlib.org) / [Paper](https://arxiv.org/pdf/1701.03757.pdf)
- [pyMC3 for Bayesian Modeling](https://github.com/pymc-devs/pymc3)
  - [Python/PyMC3 versions of the programs described in Doing bayesian data analysis by John K. Kruschke](https://github.com/aloctavodia/Doing_bayesian_data_analysis) 
- [Bayesian Modeling in Python](https://github.com/markdregan/Bayesian-Modelling-in-Python)
- [Monte Carlo method intro Video, Iain Murray NIPS2015](http://research.microsoft.com/apps/video/default.aspx?id=259575&l=i)
- [MCMC Sampling intro/tutorial](http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/)
- [Online Book on Probabilistic Generative Models, Goodman, Tenenbaum](https://probmods.org)
- [Intro/Intuition on Variational Inference (e.g. KLqp vs. KLpq dist.div.) D.MacKay Lecture](http://videolectures.net/mackay_course_14/)

#### Gaussian Processes
- [Neil Lawrence Lectures/Notebooks](http://gpss.cc/)
  - http://nbviewer.jupyter.org/github/gpschool/gprs15a/blob/master/gaussian%20process%20introduction.ipynb

### NN
- [Harnessing Deep Neural Networks with Logic Rules](http://arxiv.org/abs/1603.06318)
- [Zero-Shot Learning for Semantic Utterance
Classification - 2015 Dauphin et al.](http://arxiv.org/pdf/1401.0509.pdf)

#### Introductions to Convolutional neural networks
- [CNN Stanford introduction](http://cs231n.github.io/convolutional-networks/#conv)
- [colah's blog on convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)
- [WILDML blog on CNN for NLP problems](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

#### RNN / Attention
- [Neural Machine Translation - seq2seq, ICLR 2015 Bahdanau, Cho, Bengio](http://arxiv.org/abs/1409.0473v6)
- [Grammar as a foreign language, 2014 Vinyals, Kaiser, Koo, Petrov, Sutskever, Hinton](http://arxiv.org/abs/1412.7449)
- [Pointer Networks, 2015 Vinyals, Fortunato, Jaitly](http://arxiv.org/abs/1506.03134)
- [Neural GPUS Learn Algortihms - 2016 Kaiser, Sutskever](http://arxiv.org/pdf/1511.08228v3.pdf) (Multilayer CGRU - generalizing to 2000bit length ops w/o error)

### Generative / GAN
- [Video Frame Generation via Cross Convolution](https://github.com/tensorflow/models/tree/master/next_frame_prediction)
- [OpenAI: Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476.pdf)
- [Unsup. feature disentangling - InfoGAN](https://github.com/openai/InfoGAN)

### Reinforcement Learning
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)
- [DeepMind paper on deep RL](http://www.readcube.com/articles/10.1038%2Fnature14236?shared_access_token=Lo_2hFdW4MuqEcF3CVBZm9RgN0jAjWel9jnR3ZoTv0P5kedCCNjz3FJ2FhQCgXkApOr3ZSsJAldp-tw3IWgTseRnLpAc9xQq-vTA2Z5Ji9lg16_WvCy4SaOgpK5XXA6ecqo8d8J7l4EJsdjwai53GqKt-7JuioG0r3iV67MQIro74l6IxvmcVNKBgOwiMGi8U0izJStLpmQp6Vmi_8Lw_A%3D%3D) (Deep Q-Learning, experience replay) 
- [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)

### NLP / Q&A / Memory Networks
- [Predicting distributions with Linearizing Belief Networks](http://arxiv.org/abs/1511.05622)
- [Weakly supervised memory networks, 2015 Sukhbaatar, Szlam, Weston, Fergus](http://arxiv.org/abs/1503.08895)

### MISC
- [Stanford Unsupervised Feature Learning and Deep Learning](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial) 
  - [Backpropagation and its vectorization](http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm)
  - [Gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)
  - [Network initialization/activations explained, Stanford CS231nWinter, Karpathy](https://www.youtube.com/watch?v=gYpoJMlgyXA)
- [NIPS2014 Workshop list/videos](https://nips.cc/Conferences/2014/Schedule?type=Workshop)
- [Online Book D.McKay: Information Theory,Inference,and Learning Algorithms](http://www.inference.phy.cam.ac.uk/itprnn/book.pdf)
- [What is the intuition behind a beta distribution](http://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution)
- [sugartensor lib on top of tensorflow](https://github.com/buriburisuri/sugartensor) (e.g. [Quasi-RNN reference impl](https://github.com/Kyubyong/quasi-rnn))
- [Python](http://www.cs.ubc.ca/~nando/540-2013/python.html)


## Datasets
- [Stanford Qestion&Answering Dataset SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

## Other Books / Tutorials
- [Probability Theory - The Logic of Science, E.T. Jaynes](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf)

## Blogs
- [colah's blog](http://colah.github.io/)
  - [intuition on information/entropy/KL divergence](http://colah.github.io/posts/2015-09-Visual-Information)
- [InFERENCe](http://www.inference.vc/)
