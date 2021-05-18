# Neural Active Inference (Simplified)
This repo contains the proposed QAI Model, which is an active inference agent based on an artificial neural network trained via backpropagation of errors (backprop). This model embodies a key assumptions:
1) A simplified model is sufficient for reasonably-sized state spaces (like Mountain Car, Cartpole, etc.) -- thus this model only jointly adapts a transition model and an expected free energy (EFE) model at each time step.
2) We only need the simple Q-learning bootstrap principle to train this system (as opposed to policy gradients)
3) We normalize the instrumental and/or epistemic (scalar) signals according to the dynamic normalization scheme proposed in Ororbia & Mali (2021).

To run the code, you can use the following Bash commands:<br>
<code>
$ python train_prior.py --cfg_fname=fit_mcar_prior.cfg --gpu_id=0  # this fits/trains the local prior model to be used in the active inference agent (if a local prior is desired) 
</code><br>
<code>
$ python train_agent.py --cfg_fname=run_mcar_ai.cfg --gpu_id=0  # this trains the agent according to whatever is configured inside the *.cfg file
</code>

Inside the training configuration file, you can choose to use an alternative prior preference as follows:
1) Set <pre>instru_term = prior_local</pre> to use a local prior preference (requires running train_prior.py and storing the prior in the correct folder is used)
2) Set <pre>instru_term = prior_global</pre> to use a global hard-coded prior preference
3) Set <pre>instru_term = prior_reward</pre> to use the environment's reward signal as the prior preference (due to Complete Class Theorem)
