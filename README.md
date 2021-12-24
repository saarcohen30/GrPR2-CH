# Level-k Graph Probabilistic Recursive Reasoning (GrPR2-L) and Cognitive Hierarchy Graph Probabilistic Reasoning (GrPR2-CH)
Code for implementation of Cognitive Hierarchy Graph Probabilistic Recursive Reasoning (**<em>GrPR2-A</em>**), in the context of <em>The Cooperative Navigation Task</em> of the [Particle World environment](https://github.com/openai/multiagent-particle-envs). Specifically, GrPR2-A constitutes an extension of [FlowComm](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p456.pdf), supporting level-1 recursive reasoning <em>with communication</em>. 
If any part of this code is used, the following paper must be cited: 

**Saar Cohen and Noa Agmon. Optimizing Multi-Agent Coordination via Hierarchical Graph Probabilistic Recursive Reasoning. <em>In AAMAS'22: Proceedings of the 21th International Conference on Autonomous Agents and Multiagent Systems, 2022</em> (to appear).**

## Dependencies
Evaluations were conducted using a 12GB NVIDIA Tesla K80 GPU, and implemented in Python3 with:
- PyTorch v2.6.0 (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-a-colab)).
- PyTorch v1.12.0, which is suitable for environment without support of higher versions of PyTorch (The implementation appears [here](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-a)).

## The Cooperative Navigation Task
In this task of the Particle World environment, `n` agents must cooperate through physical actions to reach a set of $n$ landmarks. Agents observe the relative positions of nearest agents and landmarks, and are collectively rewarded based on the proximity of any agent to each landmark. In other words, the agents have to "cover" all of the landmarks. Further, the agents occupy significant physical space and are penalized when colliding with each other. Our agents learn to infer the landmark they must cover, and move there while avoiding other agents. Though the environment holds a continuous state space, agents' actions space is discrete, and given by all possible directions of movement for each agent `{up, down, left, right, stay}`. Given an interaction graph, we augment this task for enabling local information sharing between neighbors, as outlined subsequently.

## Concept
As mentioned earlier, [FlowComm](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p456.pdf) constitutes the baseline for GrPR2-A. FlowComm embeds a graph reasoning policy into [MAAC](http://proceedings.mlr.press/v97/iqbal19a.html), which trains decentralized policies in multiagent settings, using <em>centrally</em> computed critics that share an attention mechanism, selecting relevant information for each agent. The relevant files which correspond to the core of our implmentation are listed as follows:
- [`grpr2-a/utils/policies.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a/utils/policies.py) and [`grpr2-a-colab/utils/policies.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a-colab/utils/policies.py) - The class `DiscreteConditionalPolicy` implements a discrete policy, which is conditional on the actions of an agnet's opponents.
- [`grpr2-a/utils/agents.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a/utils/agents.py) and [`grpr2-a-colab/utils/agents.py`](https://github.com/saarcohen30/GrPR2-A/blob/main/grpr2-a-colab/utils/agents.py) -- The class `AttentionREGMAAgent` implements the GrPR2-A agent, which incorporates both an agent's policy and its opponents' (approximated) conditional policies.

## Execution
The [`grpr2-a/`](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-a) and [`grpr2-a-colab/`](https://github.com/saarcohen30/GrPR2-A/tree/main/grpr2-a-colab) sub-directories consist of the `regma.py` module, whose execution performs the required testbed. Specifically, the following executions are possible:
- `python regma/regma.py simple_spread_local maac` - For a setup of `n=4` agents.
- `python regma/regma.py simple_spread_hetero maac` - For the extension of the cooperative navigation task to a <em>heterogeneous<\em> communication task. That is, for `n=8`, the third agent has a larger field of vision compared to the other agents. This allows us to simulate heterogeneity in hierarchical level of thinking, as agent `3` becomes more sophisticated compared to other agents.
