# Neural Active Inference (Simplified)
This repo contains the proposed QAI Model, which is an active inference agent based on an artificial neural network trained via backpropagation of errors (backprop). This model embodies a key assumptions:
1) A simplified model is sufficient for reasonably-sized state spaces (like Mountain Car, Cartpole, etc.) -- thus this model only jointly adapts a transition model and an expected free energy (EFE) model at each time step.
2) We only need the simple Q-learning bootstrap principle to train this system (as opposed to policy gradients)
3) We normalize the instrumental and/or epistemic (scalar) signals according to the dynamic normalization scheme proposed in Ororbia & Mali (2021).
