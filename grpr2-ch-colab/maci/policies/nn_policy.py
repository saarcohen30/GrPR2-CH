import tensorflow as tf

from maci.core.serializable import Serializable
from maci.misc.overrides import overrides
from maci.policies.base import Policy


class NNPolicy(Policy, Serializable):
    def __init__(self, env_spec, obs_pl, action, scope_name=None):
        Serializable.quick_init(self, locals())

        self._observation_ph = obs_pl
        self._action = action
        self._scope_name = (tf.get_variable_scope().name
                            if not scope_name else scope_name)
        super(NNPolicy, self).__init__(env_spec)

    @overrides
    def get_action(self, observation):
        return self.get_actions(observation[None])[0], None

    @overrides
    def get_actions(self, observations):
        feeds = {self._observation_ph: observations}
        actions = tf.compat.v1.get_default_session().run(self._action, feeds)
        return actions

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def get_params_internal(self, **tags):
        # TODO: rewrite this using tensorflow collections
        if tags:
            scope = tags['scope']
        else:
            scope = self._scope_name
            # Add "/" to 'scope' unless it's empty (otherwise get_collection will
            # return all parameters that start with 'scope'.
            scope = scope if scope == '' else scope + '/'

        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)