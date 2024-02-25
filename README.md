# Neural Active Inference
This repository holds the code for paper *A Neural Active Inference Model of Perceptual-Motor Learning*, which is published on *Frontiers in Computational Neuroscience*, 2023. In this work, we propose an active inference (AI) agent based on an artificial neural network trained via backpropagation of errors (backprop). This model embodies a key assumptions:
1) A simplified model is sufficient for reasonably-sized state spaces (like Mountain Car, Cartpole, etc.) -- thus this model only jointly adapts a transition model and an expected free energy (EFE) model at each time step.
2) We only need the simple Q-learning bootstrap principle to train this system (as opposed to policy gradients)
3) We simplify the Bayesian inference by assuming a uniform prior (or uninformative prior) on the parameters of our model.

To run the code, you can use the following Bash commands.<br>

To train a local prior model to expert data (imitation learning), which will be later used in the active inference agent (if a local prior is desired), then run the following command (after setting desired values in <code>fit_mcar_prior.cfg</code>):
<pre>
$ python train_prior.py --cfg_fname=fit_mcar_prior.cfg --gpu_id=0 
</pre>
To train the final AI agent according to whatever is configured inside the <code>run_mcar_ai.cfg</code> file, run the following command:
<pre>
$ python train_agent.py --cfg_fname=run_mcar_ai.cfg --gpu_id=0
</pre>

Inside the training configuration file, you can choose to use an alternative prior preference as follows:
1) To use a local prior (model), set <pre>instru_term = prior_local</pre> which also requires running train_prior.py and storing the prior in the correct folder is used. Make sure you set the <code>prior_model_save_path</code> in the config file to point to wherever you dump/save the prior model on disk.
2) To use a global, hand-coded prior <pre>instru_term = prior_global</pre> which requires changing the tensor variable <code>self.global_mu</code> inside the QAIModel (in <code>src/model/qai_model.py</code>) to a vector of encoded mean values (default is <code>None</code>.
4) To use the reward as the global prior, set <pre>instru_term = prior_reward</pre> where we justify this by appealing to the Complete Class Theorem.
