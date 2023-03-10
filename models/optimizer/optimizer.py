from torch import optim


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(Momentum, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.Momentum(learning_rate=self.learning_rate,
                             momentum=self.momentum,
                             weight_decay=self.weight_decay,
                             grad_clip=self.grad_clip,
                             parameters=train_params)
        return opt


def Adam(model,
         learning_rate=0.001,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-08,
         parameter_list=None,
         weight_decay=None,
         **kwargs):
    train_params = [
        param for param in model.parameters() if param.requires_grad is True
    ]
    opt = optim.Adam(lr=learning_rate,
                     betas=[beta1, beta2],
                     eps=epsilon,
                     weight_decay=weight_decay,
                     params=train_params)
    return opt


# class Adam(object):
#     def __init__(self,
#                  learning_rate=0.001,
#                  beta1=0.9,
#                  beta2=0.999,
#                  epsilon=1e-08,
#                  parameter_list=None,
#                  weight_decay=None,
#                 #  grad_clip=None,
#                 #  name=None,
#                 #  lazy_mode=False,
#                  **kwargs):
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.parameter_list = parameter_list
#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         # self.grad_clip = grad_clip
#         # self.name = name
#         # self.lazy_mode = lazy_mode

#     def __call__(self, model):
#         train_params = [
#             param for param in model.parameters() if param.requires_grad is True
#         ]
#         opt = optim.Adam(
#             lr=self.learning_rate,
#             betas=[self.beta1,self.beta2],
#             eps=self.epsilon,
#             weight_decay=self.weight_decay,
#             # grad_clip=self.grad_clip,
#             # name=self.name,
#             # lazy_mode=self.lazy_mode,
#             params=train_params)
#         return opt


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        rho (float) - rho value in equation.
        epsilon (float) - avoid division by zero, default is 1e-6.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 **args):
        super(RMSProp, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.RMSProp(learning_rate=self.learning_rate,
                            momentum=self.momentum,
                            rho=self.rho,
                            epsilon=self.epsilon,
                            weight_decay=self.weight_decay,
                            grad_clip=self.grad_clip,
                            parameters=train_params)
        return opt


class Adadelta(object):

    def __init__(self,
                 learning_rate=0.001,
                 epsilon=1e-08,
                 rho=0.95,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 **kwargs):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.rho = rho
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name

    def __call__(self, model):
        train_params = [
            param for param in model.parameters() if param.trainable is True
        ]
        opt = optim.Adadelta(learning_rate=self.learning_rate,
                             epsilon=self.epsilon,
                             rho=self.rho,
                             weight_decay=self.weight_decay,
                             grad_clip=self.grad_clip,
                             name=self.name,
                             parameters=train_params)
        return opt


# class AdamW(object):

#     def __init__(self,
#                  learning_rate=0.001,
#                  beta1=0.9,
#                  beta2=0.999,
#                  epsilon=1e-8,
#                  weight_decay=0.01,
#                  multi_precision=False,
#                  grad_clip=None,
#                  no_weight_decay_name=None,
#                  one_dim_param_no_weight_decay=False,
#                  name=None,
#                  lazy_mode=False,
#                  **args):
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.epsilon = epsilon
#         self.grad_clip = grad_clip
#         self.weight_decay = 0.01 if weight_decay is None else weight_decay
#         self.grad_clip = grad_clip
#         self.name = name
#         self.lazy_mode = lazy_mode
#         self.multi_precision = multi_precision
#         self.no_weight_decay_name_list = no_weight_decay_name.split(
#         ) if no_weight_decay_name else []
#         self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

#     def __call__(self, model):
#         parameters = [
#             param for param in model.parameters() if param.trainable is True
#         ]

#         self.no_weight_decay_param_name_list = [
#             p.name for n, p in model.named_parameters()
#             if any(nd in n for nd in self.no_weight_decay_name_list)
#         ]

#         if self.one_dim_param_no_weight_decay:
#             self.no_weight_decay_param_name_list += [
#                 p.name for n, p in model.named_parameters()
#                 if len(p.shape) == 1
#             ]

#         opt = optim.AdamW(learning_rate=self.learning_rate,
#                           beta1=self.beta1,
#                           beta2=self.beta2,
#                           epsilon=self.epsilon,
#                           parameters=parameters,
#                           weight_decay=self.weight_decay,
#                           multi_precision=self.multi_precision,
#                           grad_clip=self.grad_clip,
#                           name=self.name,
#                           lazy_mode=self.lazy_mode,
#                           apply_decay_param_fun=self._apply_decay_param_fun)
#         return opt

#     def _apply_decay_param_fun(self, name):
#         return name not in self.no_weight_decay_param_name_list



def AdamW(model,learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=0.01,
                #  multi_precision=False,
                #  grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                #  name=None,
                #  lazy_mode=False,
                 **args):
        # weight_decay = 0.01 if weight_decay is None else weight_decay
        # no_weight_decay_name_list = no_weight_decay_name.split(
        # ) if no_weight_decay_name else []

        parameters = [
            param for param in model.parameters() if param.requires_grad is True
        ]

        # no_weight_decay_param_name_list = [
        #     p.name for n, p in model.named_parameters()
        #     if any(nd in n for nd in no_weight_decay_name_list)
        # ]

        # if one_dim_param_no_weight_decay:
        #     no_weight_decay_param_name_list += [
        #         p.name for n, p in model.named_parameters()
        #         if len(p.shape) == 1
        #     ]

        opt = optim.AdamW(lr=learning_rate,
                          betas=[beta1, beta2],
                          eps=epsilon,
                          params=parameters,
                          weight_decay=weight_decay,

                        #   apply_decay_param_fun=_apply_decay_param_fun
                          )
        return opt

    # def _apply_decay_param_fun(self, name):
    #     return name not in self.no_weight_decay_param_name_list
