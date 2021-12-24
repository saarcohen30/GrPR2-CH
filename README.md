# Hierarchical Graph Probabilistic Recursive Reasoning
Code for implementation of the following Hierarchical Graph Probabilistic Recursive Reasoning frameworks (in the context of <em>The Cooperative Navigation Task</em> of the [Particle World environment](https://github.com/openai/multiagent-particle-envs)):
- Level-k Graph Probabilistic Recursive Reasoning (**<em>GrPR2-L</em>**), where agent `i` at level `k` assumes that other agents are at level `k-1` and then best responds by integrating over all possible interactions induced by the interaction graph and best responses from lower-level agents to agent `i` of level `k-2`.
- Cognitive Hierarchy Graph Probabilistic Recursive Reasoning (**<em>GrPR2-CH</em>**), which lets each level-`k` player best respond to a <em>mixture</em> of strictly lower levels in the hierarchy, induced by truncation up to level `k - 1` from the underlying level distribution.

If any part of this code is used, the following paper must be cited: 

**Saar Cohen and Noa Agmon. Optimizing Multi-Agent Coordination via Hierarchical Graph Probabilistic Recursive Reasoning. <em>In AAMAS'22: Proceedings of the 21th International Conference on Autonomous Agents and Multiagent Systems, 2022</em> (to appear).**

## Dependencies
Evaluations were conducted using a 12GB NVIDIA Tesla K80 GPU, and implemented in Python3 with:
- PyTorch v2.6.0 (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-ch-colab)).
- PyTorch v1.12.0, which is suitable for environment without support of higher versions of PyTorch (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-ch)).

**Note:** Each framework contains a `requirements.txt` file, which specifies the modules required for its execution on the respective PyTorch version. For instance, for the imlementation suitable for PyTorch v2.6.0, the `requirements.txt` file cotains a script which aims at downloading all the modules required for its execution on Google's Colaboratory.

## The Cooperative Navigation Task
In this task of the Particle World environment, `n` agents must cooperate through physical actions to reach a set of $n$ landmarks. Agents observe the relative positions of nearest agents and landmarks, and are collectively rewarded based on the proximity of any agent to each landmark. In other words, the agents have to "cover" all of the landmarks. Further, the agents occupy significant physical space and are penalized when colliding with each other. Our agents learn to infer the landmark they must cover, and move there while avoiding other agents. Though the environment holds a continuous state space, agents' actions space is discrete, and given by all possible directions of movement for each agent `{up, down, left, right, stay}`. Given an interaction graph, we augment this task for enabling local information sharing between neighbors, as outlined subsequently.

## Concept
As mentioned earlier, [FlowComm](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p456.pdf) constitutes the baseline for GrPR2-A. FlowComm embeds a graph reasoning policy into [MAAC](http://proceedings.mlr.press/v97/iqbal19a.html), which trains decentralized policies in multiagent settings, using <em>centrally</em> computed critics that share an attention mechanism, selecting relevant information for each agent. The relevant files which correspond to the core of our implmentation are listed as follows:
- [`grpr2-a/utils/policies.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a/utils/policies.py) and [`grpr2-a-colab/utils/policies.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a-colab/utils/policies.py) - The class `DiscreteConditionalPolicy` implements a discrete policy, which is conditional on the actions of an agnet's opponents.
- [`grpr2-a/utils/agents.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a/utils/agents.py) and [`grpr2-a-colab/utils/agents.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a-colab/utils/agents.py) -- The class `AttentionREGMAAgent` implements the GrPR2-A agent, which incorporates both an agent's policy and its opponents' (approximated) conditional policies.

## Execution
The [`grpr2-ch/`](https://github.com/saarcohen30/GrPR2-CH/tree/main/grpr2-ch) and [`grpr2-ch-colab/`](https://github.com/saarcohen30/GrPR2-CH/tree/main/grpr2-ch-colab) sub-directories consist of the `main.py` module, whose execution performs the required testbed. Specifically, the following executions are possible:
- `python grpr2-ch/main.py simple_spread_local maac` or `python grpr2-ch-colab/main.py simple_spread_local maac --train_graph True` - For a setup of `n=4` agents.
- `python regma/regma.py simple_spread_hetero maac` or `python grpr2-ch-colab/main.py simple_spread_hetero maac` - For a setup of `n=8` agents.

### Important Flags
- `--train_graph` -- In both setups (of either `n=4` or `n=8` agents), one can possibly decide whether to train the graph reasoning policy or not. After specifying `--train_graph true` upon the execution, the graph reasoning policy will be trained. By default, the graph reasoning policy will **not** be trained.
- `--pretrained_graph` -- In both setups (of either `n=4` or `n=8` agents), one can possibly decide whether to utilize a pre-trained graph reasoning policy, which shall be stored in a file named as `local_graph.pt`. For this sake, the `--pretrained_graph` flag shall be set to true by specifying `--pretrained_graph true` upon the execution. By default, a pretrained graph reasoning policy will **not** be incorporated.
- `--model_names_setting` - This flag specifies the names of the model to be trained. The possible models for initializing an self-play enviroment are as follows:

| The Flag's Argument | The Model's Description |
| ------------- | ------------- |
| GrPR2AC`k`_GrPR2AC`k`  | For level-`k` GrPR2-CH agents |
| PR2AC`k`_PR2AC`k`  | For level-`k` GrPR2-L agents |
| DDPG_DDPG | For DDPG independent learners, which are regarded as having level-0 reasoning. |
| MADDPG_MADDPG | For MADDPG agents, which are regarded as having level-0 reasoning. |
| DDPG-ToM_DDPG-ToM | For DDPG agents with a level-1 [Theory-of-Mind model](http://proceedings.mlr.press/v80/rabinowitz18a/rabinowitz18a.pdf) that captures the dependency of an agent’s policy on opponents’ mental states (DDPG-ToM). |
| DDPG-OM_DDPG-OM | For DDPG agents with a level-0 model of [opponent modeling](http://proceedings.mlr.press/v48/he16.pdf), which is implemented by augmenting DDPG with an opponent module (DDPG-OM) that predicts opponents' behaviors in future states. |
