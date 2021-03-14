## Corai

![CorAI](CorAI.png)



## Team and project overview 

<details open="open">
<summary><h3 style="display: inline-block">Team presentation</h2></summary>
<p>We are a team of 5 students from Telecom SudParis. Eager to learn more about Artificial Intelligence, we saw the Hi!ckathon as an opportunity. We initially divided the work between</p>
<ul>
<li>the QLearning team</li>
<li>the Double QLearning team</li>
<li>the Deep QLearning team</li>
</ul>
<p>and from Saturday between</p>
<ul>
<li>the optimization team</li>
<li>the video team</li>
<li>the report team</li>
</ul>
</details>


<details open="open">
<summary><h3 style="display: inline-block">General strategy</h2></summary>
Our approach consists in using an autonomous agent to make the most optimal decision based on its current state and observations. The agent can then decide to charge the battery from the PV, discharge the battery, import from the grid, export to the grid or charge the battery from the grid. We chose to use Reinforcement Learning to perform this task.
</details>


## Scientific approach

<details open="open">
<summary><h3 style="display: inline-block">Approach description</h2></summary>
<p>&nbsp;The subject of reinforcement learning was absolutely new to all of us (we are used to working on  supervised/unsupervised learning or latent data models). After 5 hours of learning (standford courses, youtube, …) we decided to build an agent based on expensive algorithms (QLearning, SARSA, Double QLearning, Double SARSA, DeepQLearning) and then build decisions rules based on their intermediate quantities. We were inspired by various web resources such as <a href="http://towardsdatascience.com/">Towards Data Science</a>, <a href="https://medium.com/">Medium</a> and publications from <a href="https://hal.archives-ouvertes.fr/">HAL Ouvertes</a>.</p>
<p>&nbsp;We started with the QLearning (which was the closest to our tiny theoretical knowledge). These approaches were a disaster : programming issues, unappropriated learning rate decays, slow convergence, ... So we decided to investigate DeepQLearning. After hours of research, we had multiple “half working” implementations using random (and sometimes old) libraries which were unexploitable.</p>
<p>&nbsp;On Sunday evening, we decided to split : one team would play with SARSA while the other would play with DeepQLearning (<em>keras-rl2</em> as we weren’t yet aware of the existence of <em>tensorforce</em> or <em>tf_agents</em>). SARSA is especialy relevant because of our frugality constrains and is a step in the right direction compared to <a href="https://hal.archives-ouvertes.fr/hal-02382232">this paper</a> which was the base inspiration of our agent.<br>&nbsp;The SARSA team was very successful and developed a custom algorithm with the following features :</p>
<ul>
<li>custom learning rate decay for alpha based</li>
<li>exponential learning rate decay for epsilon
some “in between” version of “Double SARSA” where two Q matrices are maintained in parallel (this was originally a programmation mistake but we modified it towards an efficient method)</li>
<li>Q-transfer learning</li>
<li>custom value for Q initialization</li>
</ul>
<p>&nbsp;We then asked ourselves &quot;<em>What about the states that weren’t met during training ?</em>&quot; and decided to couple this algorithm with a decision tree classifier (an idea also found <a href="https://hal.archives-ouvertes.fr/hal-02382232">here</a>)! This classifier will make a better guess than just repeating the initial value. This approach had two positive impacts</p>
<ul>
<li>we can perform the inference on never seen states</li>
<li>we can use the decision tree to hint possible rules for a rule-based algorithm (the optimal maximum depth of those trees appeared to be pretty small which is surprising considering the complexity of the problem)</li>
</ul>
<p>&nbsp;After hours of hyperparameters tuning between the “simple” and the “double” models, the latter method turned out to have a very satisfying profitability with an impressive frugality.</p>
<p>&nbsp;Regarding the DeepQLearning lead, we tried various neural networks with only dense layers. After hyperparameters tuning (number of hidden layers, size of layers, learning rates, …), this lead turned out to be way slower than the previously mentioned ones for a similar profitability. After a day of work, we decided to drop this lead.</p>


</details>

<details open="open">
<summary><h3 style="display: inline-block">Future improvements</h2></summary>
<p>One avenue that we did not have time to explore would have been to build a latent dataset and construct a hidden markov chain model to observe much more useful information underneath the raw data to predict the state of energy consumption. We would have used the Expectation-Maximisation algorithm to explore the raw data and estimate a transition matrix and apply the Baulm-Welch algorithm to find out the hidden states and forecast the energy consumption.</p>
<p>As we said, we couple our double SARSA with a decision tree classifier. We could have tested other classifiers to perform the inference on never seen states like random forest or support vector machines.</p></details>




## Project usage

The notebook we use is `evaluation_notebook.ipynb` , it only uses `pymgrid` and `sklearn` with the `DiscreteEnvironment`

Here is a quick walk-through our notebook

* The project dependencies are loaded and the building are loaded as well
* `make_hyperparameters` build a dictionary which contains the hyper-parameters for the model
  * `C` is the initial value of `Q`
  * `alpha` is the learning rate in the Bellman equation
  * `omega` is the exponent in the `alpha`-decay rule
  * `discount_factor` is the discount in Bellman equation
  * `epsilon` is the epsilon initial value in the QLearning algorithm. `epsilon_min` is its minimal value, `expsilon_decay` is the parameter of the decay function (exponential is `epsilon_expo` otherwise it's linear)
  * `train_days` number of days used (Q-Transfer)
  * `train_episodes` number of episodes used
  * `train_episodes_decay` episode decay (Q-Transfer)
* `test` uses only the output of the SARSA algorithm (Q matrices) to test the model
* `test_ml` integrates our classifier for testing the model
* `train` estimates the Q matrices using the parameters in `p` and an environment `env`
* `build_clf` builds the decision tree (and its dataset) based on Q matrices
* `benchmark` performs the full benchmark with parameter `p`

Running all the cells of the notebook performs the benchmark with our optimal parameters.

