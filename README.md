## Using this template

This template specifies the minimum information to provide to the jury to support your project. You are free to go beyond these mandatory sections to include additional explanations.



## Team and project overview 

<details open="open">
<summary><h3 style="display: inline-block">Team presentation</h2></summary>
<p>We are a team of 5 students from Telecom SudParis. Eager to learn more about Artificial Intelligence, we saw the Hi!ckathon as an opportunity. We initially divided the work between</p>
<ul>
<li>the QLearning team</li>
<li>the Double QLearning team</li>
<li>the Deep QLearning team</li>
</ul>
<p>and from Sunday between</p>
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
<p>&nbsp;On Sunday evening, we decided to split : one team would play with SARSA while the other would play with DeepQLearning (<em>keras-rl2</em> as we weren’t yet aware of the existence of <em>tensorforce</em> or <em>tf_agents</em>).<br>&nbsp;The SARSA team was very successful and developed a custom algorithm with the following features :</p>
<ul>
<li>custom learning rate decay for alpha based</li>
<li>exponential learning rate decay for epsilon
some “in between” version of “Double SARA” where two Q matrices are maintained in parallel (this was originally a programmation mistake but we modified it towards an efficient method)</li>
<li>Q-transfer learning</li>
<li>custom value for Q initialization</li>
</ul>
<p>&nbsp;We then asked ourselves &quot;<em>What about the states that weren’t met during training ?</em>&quot; and decided to couple this algorithm with a decision tree classifier ! This classifier will make a better guess than just repeating the initial value. This approach had two positive impacts</p>
<ul>
<li>we can perform the inference on never seen states</li>
<li>we can use the decision tree to hint possible rules for a rule-based algorithm (the optimal maximum depth of those trees appeared to be pretty small which is surprising considering the complexity of the problem)</li>
</ul>
<p>&nbsp;After hours of hyperparameters tuning between the “simple” and the “double” models, the latter method turned out to have a very satisfying profitability with an impressive frugality.</p>
<p>&nbsp;Regarding the DeepQLearning lead, we tried various neural networks with only dense layers. After hyperparameters tuning (number of hidden layers, size of layers, learning rates, …), this lead turned out to be way slower than the previously mentioned ones for a similar profitability. After a day of work, we decided to drop this lead.</p>

</details>

<details open="open">
<summary><h3 style="display: inline-block">Future improvements</h2></summary>
<p>One avenue that we did not have time to explore would have been to build a latent dataset and construct a hidden markov chain model to observe much more useful information underneath the raw data to predict the state of energy consumption. We would have used the Expactation-Maximisation algorithm to explore the raw data and estimate a transition matrix and apply the Baulm-Welch algorithm to find out the hidden states and forecast the energy consumption.</p>
<p>As we said, we couple our double SARSA with a decision tree classifier. We could have tested other classifiers to perform the inference on never seen states like random forest or support vector machines.</p></details>



## Project usage



