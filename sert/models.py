import lasagne
import logging
import numpy as np
import scipy.sparse as sparse
import time
import theano
import theano.tensor as T
import theano.sparse as S

try:
    import theano.compile.nanguardmode as nanguardmode
except:
    import warnings
    warnings.warn('Module theano.compile.nanguardmode is not available.')

    nanguardmode = None


def check_param(variable, force_gpu=False):
    if not isinstance(variable, theano.tensor.sharedvar.TensorSharedVariable):
        logging.warning('Variable %s is of type %s '
                        '(expected TensorSharedVariable).',
                        variable, type(variable))

    if theano.config.device.startswith('gpu'):
        if isinstance(variable.type, theano.sandbox.cuda.CudaNdarrayType):
            logging.warn('Variable "%s" is already living on the GPU.',
                         variable.name)

            return variable
        elif variable.dtype == 'float32':
            if isinstance(variable.type, T.TensorType):
                if force_gpu:
                    logging.info('Forcing variable "%s" to live on the GPU.',
                                 variable.name)

                    return theano.sandbox.cuda.basic_ops.gpu_from_host(
                        variable)
                else:
                    logging.info('Variable %s lives in CPU land, but could be '
                                 'forced on the GPU by setting force_gpu '
                                 'to True.',
                                 variable.name)

                    return variable
            else:
                logging.warn('Variable "%s" is not a TensorType (actual: %s) '
                             'and is therefore forced to live on the CPU; '
                             'this might limit performance.',
                             variable.name, variable.type)

                return variable
        else:
            logging.warn('Variable "%s" is of type %s and is therefore '
                         'forced to live on the CPU; this might limit '
                         'performance.',
                         variable.name, variable.dtype)

            return variable
    else:
        logging.info('Variable "%s" will live in CPU land.', variable.name)

        return variable


def maybe_densify_variable(possibly_sparse_variable, *args, **kwargs):
    if not isinstance(possibly_sparse_variable.type, S.type.SparseType):
        return possibly_sparse_variable

    return densify_sparse_variable(
        possibly_sparse_variable,
        *args, **kwargs)


def densify_sparse_variable(sparse_variable, force_gpu):
    assert isinstance(sparse_variable.type, S.type.SparseType)

    variable = S.dense_from_sparse(sparse_variable)

    if (theano.config.device == 'gpu' and
            force_gpu and
            not isinstance(variable.type,
                           theano.sandbox.cuda.CudaNdarrayType)):
        variable = theano.sandbox.cuda.basic_ops.gpu_from_host(variable)

    logging.debug('Densified variable "%s" to "%s".',
                  sparse_variable, variable)

    return variable


def l2_regularization(objects):
    """
    Apply l2 regularization.

    Given a set of objects (Lasagne layers or Theano shared variables)
    returns an expression which l2 regularizes those objects or the
    parameters thereof.
    """
    params = []

    for o in objects:
        if isinstance(o, lasagne.layers.Layer):
            object_params = o.get_params(regularizable=True)

            if not object_params:
                logging.error(
                    'Layer %s has no regularizable parameters.', o)

            params.extend(object_params)
        elif isinstance(o, theano.compile.SharedVariable):
            params.append(o)
        else:
            raise RuntimeError(
                'Object %s of type %s is not regularizable.' %
                (o, type(o)))

    logging.debug('L2 regularizing: %s.', params)

    return sum(T.sqr(param).sum() for param in params)


def detect_floating_point_exceptions(i, node, fn):
    """
    Detect floating point exceptions.

    Monitoring function for theano.compile.MonitorMode which
    checks for non-finite real numbers (e.g. NaN or infinity)
    during execution.

    Use as follows when compiling a Theano function:

        f = theano.function(
            [x], [5 * x],
            mode=theano.compile.MonitorMode(
                post_func=theano_utils.detect_floating_point_exceptions))

    See http://deeplearning.net/software/theano/tutorial/debug_faq.html
    for more information.
    """
    detected_exception = False

    for output in fn.outputs:
        if not isinstance(output[0], np.random.RandomState) and \
                not np.isfinite(output[0]).all():
            logging.error('NaN or infinity detected.')
            logging.error(theano.printing.debugprint(node, file='str'))
            logging.error('Inputs : %s' % [input[0] for input in fn.inputs])
            logging.error('Outputs: %s' % [output[0] for output in fn.outputs])

            detected_exception = True

    if detected_exception:
        raise RuntimeError(
            'Floating point exception detected; check your logs.')


class SparseProjectionLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_representations, representation_size,
                 representations=lasagne.init.GlorotUniform(), **kwargs):
        super(SparseProjectionLayer, self).__init__(incoming, **kwargs)

        self.num_representations = num_representations
        self.representation_size = representation_size

        self.representations = check_param(
            self.add_param(representations,
                           (num_representations, representation_size),
                           name='Representations',
                           regularizable=True))

    def get_output_shape_for(self, input_shape):
        if len(input_shape) != 2:
            raise NotImplementedError()

        return (input_shape[0], input_shape[1], self.representation_size)

    def get_output_for(self, input, *args, **kwargs):
        return self.representations[input]

    def get_representations(self):
        return self.representations.get_value(borrow=True)


class ProductTimestepsLayer(lasagne.layers.Layer):

    def __init__(self, incoming, renormalize, **kwargs):
        super(ProductTimestepsLayer, self).__init__(incoming, **kwargs)

        self.renormalize = renormalize

    def get_output_shape_for(self, input_shape):
        if len(input_shape) != 3:
            raise NotImplementedError()

        return (input_shape[0], input_shape[2])

    def get_output_for(self, input, *args, **kwargs):
        log_probabilities = T.log(T.clip(input, 1e-7, 1.0 - 1e-7))
        joint_log_probabilities = log_probabilities.sum(axis=1)

        assert joint_log_probabilities.ndim == 2, \
            'Expected tensor of second order, ' \
            'but received {0}-order tensor.'.format(
                joint_log_probabilities.ndim)

        # Bring back to real-space and normalize.
        if self.renormalize:
            return T.nnet.softmax(joint_log_probabilities)
        else:
            return T.exp(joint_log_probabilities)


class MeanLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axis, **kwargs):
        super(MeanLayer, self).__init__(incoming, **kwargs)

        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis + 1:]

    def get_output_for(self, input, *args, **kwargs):
        return input.mean(axis=self.axis)


class ClipLayer(lasagne.layers.Layer):

    def __init__(self, incoming, lower_bound, upper_bound, **kwargs):
        super(ClipLayer, self).__init__(incoming, **kwargs)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return input.clip(self.lower_bound, self.upper_bound)


class WeightedObjective(object):
    _valid_aggregation = set([None, 'mean', 'sum'])

    def __init__(self,
                 input_layer,
                 loss_function=lasagne.objectives.squared_error,
                 aggregation='mean'):
        self.input_layer = input_layer
        self.loss_function = loss_function
        self.target_var = T.matrix("target")
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        self.aggregation = aggregation

    def get_loss(self,
                 weights=None,
                 input=None,
                 target=None,
                 aggregation=None,
                 **kwargs):
        network_output = lasagne.layers.get_output(
            self.input_layer, input, **kwargs)

        if target is None:
            target = self.target_var
        if aggregation not in self._valid_aggregation:
            raise ValueError('aggregation must be \'mean\', \'sum\', '
                             'or None, not {0}'.format(aggregation))
        if aggregation is None:
            aggregation = self.aggregation

        losses = self.loss_function(network_output, target, input=input)

        if weights is not None:
            losses *= weights

        if aggregation is None or aggregation == 'mean':
            return losses.mean()
        elif aggregation == 'sum':
            return losses.sum()
        else:
            raise RuntimeError('This should have been caught earlier')


def clipped_categorical_crossentropy(output, target):
    clipped_output = T.clip(output, 1e-7, 1.0 - 1e-7)

    return T.nnet.categorical_crossentropy(clipped_output, target)


class ModelInterface(object):

    # Injects verification code in the compute graph for detecting
    # floating point exceptions.
    #
    # Prints information about nodes that generate these exceptions
    # and terminates computation accoridingly by raising a RuntimeError.
    #
    # Valid values: False, 'cpu' or 'theano'
    __DETECT_EXCEPTIONS__ = False

    assert __DETECT_EXCEPTIONS__ in (False, 'cpu', 'theano')

    if __DETECT_EXCEPTIONS__ == 'theano':
        assert nanguardmode is not None, \
            'NanGuardMode is not available.'

    TRAIN, VALIDATE, TEST = range(2, 5)

    def __init__(self, batch_size):
        assert batch_size > 0

        self.batch_size = batch_size

        logging.debug('Batch size: %d', self.batch_size)

    @classmethod
    def _get_batch_slice(cls, batch_index, batch_size):
        start = batch_index * batch_size
        end = (batch_index + 1) * batch_size

        return slice(start, end)

    def _create_param(self, variable, name):
        return check_param(
            theano.shared(variable, name, borrow=True))

    def _get_theano_monitor_mode(self):
        mode = None

        if self.__DETECT_EXCEPTIONS__ == 'cpu':
            mode = theano.compile.MonitorMode(
                post_func=detect_floating_point_exceptions)
        elif self.__DETECT_EXCEPTIONS__ == 'theano':
            mode = nanguardmode.NanGuardMode(
                nan_is_error=True,
                inf_is_error=True,
                big_is_error=False)

        logging.debug('Injection mode: %s.', repr(mode))

        return mode

    def _number_of_batches(self, num_instances):
        return num_instances // self.batch_size

    def _iterate_batches(self, fn, num_instances,
                         report_interval=10000, shuffle=False):
        start = time.time()

        num_batches = self._number_of_batches(num_instances)
        incomplete_batch_size = num_instances % self.batch_size
        if incomplete_batch_size > 0:
            logging.warning('\tIgnoring incomplete batch of size %d.',
                            incomplete_batch_size)

        results = []

        batch_indices = list(range(num_batches))
        if shuffle:
            logging.debug('Shuffling batches.')

            np.random.shuffle(batch_indices)

        for batch_idx in batch_indices:
            results.append(fn(batch_idx))

            if not np.all(np.isfinite(results[-1])):
                raise RuntimeError(
                    'Encountered NaN or infinity ({error}) '
                    'during batch iteration '
                    '(batch {batches_finished}/{num_batches}).'.format(
                        error=results[-1],
                        batches_finished=len(results),
                        num_batches=len(batch_indices)))

            if results and (len(results) % report_interval == 0 or
                            len(results) == num_batches):
                time_since_measure_start = float(time.time() - start)
                batches_per_second = len(results) / time_since_measure_start

                remaining_batches = num_batches - len(results)
                estimated_remaining_seconds = (
                    remaining_batches / batches_per_second)

                minutes_remaining = estimated_remaining_seconds / 60
                seconds_remaining = estimated_remaining_seconds % 60

                logging.info(
                    '\tProcessed %d batches; %.2f batches per second; '
                    '%d minutes %d seconds remaining.',
                    len(results), batches_per_second,
                    minutes_remaining, seconds_remaining)

        return num_batches, results

    def train(self):
        raise NotImplementedError()

    def train_error(self):
        raise NotImplementedError()

    def validation_error(self):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()


class ModelBase(ModelInterface):

    # Outputs all intermediate results in the compute graph.
    #
    # Very resource-intensive and only useful when working
    # with small toy example data sets.
    __DEBUG__ = False

    def __init__(self, batch_size,
                 training_set, validation_set,
                 learning_method):
        super(ModelBase, self).__init__(batch_size)

        if self.__DEBUG__:
            logging.warning('Verbosely outputting all intermediate results; '
                            'this will severly limit performance and make '
                            'your logs blow up.')

        if self.__DETECT_EXCEPTIONS__:
            logging.warning('Actively checking for floating point exceptions; '
                            'this might limit performance, especially if you '
                            'are running on GPUs (%s).',
                            self.__DETECT_EXCEPTIONS__)

        self.learning_method = learning_method

        self.training_num_instances = training_set[1].shape[0]
        self.validation_num_instances = validation_set[1].shape[0]

        # Determine number of instance features.
        self.num_instance_features = np.prod(training_set[0].shape[1:])

        assert self.num_instance_features == training_set[0].shape[1]

        if np.prod(validation_set[0].shape):
            assert self.num_instance_features == validation_set[0].shape[1]
        else:
            validation_set = (
                validation_set[0].reshape(
                    0, self.num_instance_features),
                validation_set[1])

        logging.info('Data set contains %d training instances '
                     'and %d validation instances',
                     self.training_num_instances,
                     self.validation_num_instances)

        assert training_set[0].dtype == validation_set[0].dtype
        assert training_set[1].dtype == validation_set[1].dtype

        self.input_dtype = training_set[0].dtype
        self.output_dtype = training_set[1].dtype

        self.training_set = training_set
        self.validation_set = validation_set

        self.x_train = self._create_param(
            training_set[0], 'X-Train')
        self.y_train = self._create_param(
            training_set[1], 'Y-Train')
        self.w_train = self._create_param(
            training_set[2], 'W-Train')

        self.x_validate = self._create_param(
            validation_set[0], 'X-Validation')
        self.y_validate = self._create_param(
            validation_set[1], 'Y-Validation')

    def _get_givens(self, function_type, batch_index,
                    symbolic_function_inputs):
        assert function_type in (ModelBase.TRAIN,
                                 ModelBase.VALIDATE,
                                 ModelBase.TEST)

        givens = {}

        if function_type == ModelBase.TRAIN:
            givens[symbolic_function_inputs['x_batch']] = \
                maybe_densify_variable(
                    self.x_train[self._get_input_batch_slice(batch_index)],
                    force_gpu=True)
            givens[symbolic_function_inputs['w_batch']] = \
                self.w_train[self._get_output_batch_slice(batch_index)]
            givens[symbolic_function_inputs['y_batch']] = \
                maybe_densify_variable(
                    self.y_train[self._get_output_batch_slice(batch_index)],
                    force_gpu=True)
        elif function_type == ModelBase.TEST:
            givens[symbolic_function_inputs['x_batch']] = \
                maybe_densify_variable(
                    self.x_train[self._get_input_batch_slice(batch_index)],
                    force_gpu=True)
            givens[symbolic_function_inputs['y_batch']] = \
                maybe_densify_variable(
                    self.y_train[self._get_output_batch_slice(batch_index)],
                    force_gpu=True)
        elif function_type == ModelBase.VALIDATE:
            givens[symbolic_function_inputs['x_batch']] = \
                maybe_densify_variable(
                    self.x_validate[self._get_input_batch_slice(batch_index)],
                    force_gpu=True)
            givens[symbolic_function_inputs['y_batch']] = \
                maybe_densify_variable(
                    self.y_validate[self._get_output_batch_slice(batch_index)],
                    force_gpu=True)
        else:
            raise NotImplementedError()

        return givens

    def _get_input_batch_slice(self, batch_index):
        return ModelBase._get_batch_slice(batch_index, self.batch_size)

    def _get_output_batch_slice(self, batch_index):
        return ModelBase._get_batch_slice(batch_index, self.batch_size)

    def _create_functions(self, output_layer, loss_train, loss_eval,
                          symbolic_function_inputs,
                          additional_params=[],
                          predict_layer=None):
        assert 'x_batch' in symbolic_function_inputs
        assert 'y_batch' in symbolic_function_inputs

        if predict_layer is None:
            predict_layer = output_layer

        batch_index = T.iscalar('batch_index')

        params = additional_params
        params.extend(lasagne.layers.get_all_params(output_layer))

        logging.info('Learning parameters: %s.', params)
        logging.info('Learning method: %s.', self.learning_method)

        updates = self.learning_method(loss_or_grads=loss_train,
                                       params=params)

        #
        # Predict function.
        #
        self.predict_fn = self._create_predict_fn(
            symbolic_function_inputs['x_batch'], predict_layer)

        # Print feed-forward function.
        logging.debug(
            'Prediction expression: %s',
            theano.printing.pprint(
                lasagne.layers.get_output(
                    predict_layer,
                    symbolic_function_inputs['x_batch'],
                    deterministic=True)))

        is_training_y_sparse = sparse.isspmatrix_csr(self.training_set[1])
        is_validation_y_sparse = sparse.isspmatrix_csr(self.validation_set[1])

        if is_training_y_sparse != is_validation_y_sparse:
            raise RuntimeError('Either training or validation truths are '
                               'sparse while the other is dense.')

        # Print update rules.
        for param, update_rule in updates.items():
            logging.debug('Parameter %s <- %s',
                          param, theano.printing.pprint(update_rule))

        #
        # Training function.
        #
        self.train_fn = theano.function(
            [batch_index], loss_train,
            updates=updates,
            givens=self._get_givens(ModelBase.TRAIN,
                                    batch_index,
                                    symbolic_function_inputs),
            mode=self._get_theano_monitor_mode(),
        )

        #
        # Test function (loss on training set).
        #
        self.test_fn = theano.function(
            [batch_index], loss_eval,
            givens=self._get_givens(ModelBase.TEST,
                                    batch_index,
                                    symbolic_function_inputs),
        )

        #
        # Validation function (loss on validation set).
        #
        self.validate_fn = theano.function(
            [batch_index], loss_eval,
            givens=self._get_givens(ModelBase.VALIDATE,
                                    batch_index,
                                    symbolic_function_inputs),
        )

        logging.debug('Verifying Theano compute graph.')

        for fn in [self.predict_fn, self.train_fn,
                   self.test_fn, self.validate_fn]:
            for node in fn.maker.fgraph.toposort():
                input_types = set(input.dtype for input in node.inputs
                                  if hasattr(input, 'dtype'))
                output_types = set(input.dtype for input in node.outputs
                                   if hasattr(input, 'dtype'))

                logging.debug(
                    'Operation %s with input types %s and output types %s.',
                    node.op, input_types, output_types)

                if 'float64' in input_types or 'float64' in output_types:
                    logging.error('Discovered double-precision floating point '
                                  'operations in compute graph.')

                    raise RuntimeError()

    def _create_predict_fn(self, x_batch, predict_layer):
        return theano.function(
            [x_batch],
            lasagne.layers.get_output(
                predict_layer,
                inputs=x_batch,
                deterministic=True))

    def train(self):
        logging.info('Training on %d training instances (%d batches).',
                     self.training_num_instances,
                     self._number_of_batches(self.training_num_instances))

        num_batches, errors = self._iterate_batches(
            self.train_fn, self.training_num_instances,
            report_interval=1000, shuffle=True)

        return num_batches, np.mean(errors)

    def train_error(self):
        logging.info('Measuring error on %d training instances (%d batches).',
                     self.training_num_instances,
                     self._number_of_batches(self.training_num_instances))

        num_batches, errors = self._iterate_batches(
            self.test_fn, self.training_num_instances)

        return np.mean(errors), np.std(errors)

    def validation_error(self):
        logging.info('Measuring error on %d validation instances '
                     '(%d batches).',
                     self.validation_num_instances,
                     self._number_of_batches(self.validation_num_instances))

        num_batches, errors = self._iterate_batches(
            self.validate_fn, self.validation_num_instances)

        return np.mean(errors), np.std(errors)

    def get_state(self):
        state = [self.predict_fn]

        all_representations = self.get_representations()
        if not hasattr(all_representations, '__iter__'):
            all_representations = (all_representations, )

        for representations in all_representations:
            state.append(representations)

        return state

    def get_representations(self):
        raise RuntimeError()


class LanguageModelBase(ModelBase):

    def __init__(self,
                 window_size,
                 representations_init,
                 regularization_lambda,
                 regularization_fn,
                 **kwargs):
        super(LanguageModelBase, self).__init__(**kwargs)

        assert window_size >= 1
        self.window_size = window_size

        self.initial_representations = representations_init

        self.vocabulary_size = representations_init.shape[0]
        self.representation_size = representations_init.shape[1]

        self.regularization_lambda = regularization_lambda

        self.regularization_fn = regularization_fn

    def _create_projection_layer(self, input_layer):
        self.projection_layer = SparseProjectionLayer(
            input_layer,
            num_representations=self.vocabulary_size,
            representation_size=self.representation_size,
            representations=self.initial_representations)

        logging.debug('Adding projection layer with '
                      'vocabulary size %d and output shape %s.',
                      self.vocabulary_size,
                      self.projection_layer.output_shape)

        return self.projection_layer

    def _finalize(self, loss_fn,
                  weight_layers, projection_layers,
                  eval_layer, predict_layer=None,
                  additional_params=[]):
        assert hasattr(weight_layers, '__iter__')
        assert hasattr(projection_layers, '__iter__')

        output_layer = eval_layer

        # Be flexible in terms of batch input formats.
        x_batch = T.matrix('input_indices', dtype=self.input_dtype)

        # Do not be flexible w.r.t. output type.
        y_batch = (
            T.ivector('y') if self.training_set[1].ndim == 1
            else T.fmatrix('y'))

        # Instance weights for training.
        w_batch = T.fvector('weights')

        objective = WeightedObjective(
            output_layer, loss_function=loss_fn)

        loss_train = objective.get_loss(
            weights=w_batch,
            input=x_batch,
            target=y_batch,
            deterministic=False)

        loss_eval = objective.get_loss(
            input=x_batch, target=y_batch, deterministic=True)

        loss_train += self._regularization(
            weight_layers, projection_layers)

        self._create_functions(output_layer, loss_train, loss_eval,
                               dict(x_batch=x_batch,
                                    y_batch=y_batch,
                                    w_batch=w_batch),
                               predict_layer=predict_layer,
                               additional_params=additional_params)

    def _regularization(self, weight_layers, projection_layers):
        acc_regularization = 0.0

        if self.regularization_lambda > 0.0:
            m = self.batch_size

            logging.debug('Adding regularization (lambda=%.10f).',
                          self.regularization_lambda)

            regularization = (
                (self.regularization_lambda *
                    self.regularization_fn(weight_layers)) /
                (2.0 * m))

            acc_regularization += regularization

        if (self.regularization_lambda > 0.0 and
           projection_layers):
            m = self.batch_size

            logging.debug(
                'Adding representation regularization (lambda=%.10f).',
                self.regularization_lambda)

            representation_regularization = (
                (self.regularization_lambda *
                 self.regularization_fn(projection_layers)) /
                (2.0 * m))

            acc_regularization += representation_regularization

        return acc_regularization

    def get_representations(self):
        if hasattr(self, 'projection_layer'):
            return self.projection_layer.get_representations()
        else:
            return None


class LanguageModel(LanguageModelBase):

    def __init__(self,
                 batch_size, window_size,
                 representations_init,
                 output_layer_size,
                 regularization_lambda,
                 training_set,
                 validation_set):
        super(LanguageModel, self).__init__(
            batch_size=batch_size,
            window_size=window_size,
            representations_init=representations_init,
            regularization_lambda=regularization_lambda,
            regularization_fn=l2_regularization,
            training_set=training_set, validation_set=validation_set,
            learning_method=lasagne.updates.adadelta)

        input_layer = lasagne.layers.InputLayer(
            shape=(self.batch_size, self.window_size))

        logging.debug('Input layer has shape %s.', input_layer.output_shape)

        projection_layer = self._create_projection_layer(input_layer)
        previous_layer = projection_layer

        new_shape = (
            self.batch_size * self.window_size,
            previous_layer.output_shape[-1])

        logging.debug('Adding reshape layer from %s to %s.',
                      previous_layer.output_shape, new_shape)

        # Reshape; for compatibility with DenseLayer.
        previous_layer = lasagne.layers.ReshapeLayer(
            previous_layer, new_shape)

        dense_nonlinearity = lasagne.nonlinearities.softmax

        logging.debug('Adding dense layer on top of %s with nonlinearity %s.',
                      previous_layer.output_shape, dense_nonlinearity)

        dense_layer = lasagne.layers.DenseLayer(
            previous_layer,
            num_units=output_layer_size,
            nonlinearity=dense_nonlinearity)

        previous_layer = dense_layer

        # Reshape again.
        previous_layer = lasagne.layers.ReshapeLayer(
            previous_layer,
            (self.batch_size, self.window_size, output_layer_size))

        logging.debug('Added %s output layer with %d units; outputs %s.',
                      dense_nonlinearity,
                      output_layer_size,
                      previous_layer.output_shape)

        eval_layer = ProductTimestepsLayer(
            previous_layer, renormalize=True)

        logging.debug('Evaluation layer of size %s.', eval_layer.output_shape)

        predict_layer = previous_layer

        logging.debug('Prediction layer of size %s.',
                      predict_layer.output_shape)

        self._finalize(
            lambda output, target, input: clipped_categorical_crossentropy(
                output, target),
            weight_layers=[dense_layer],
            projection_layers=[projection_layer],
            eval_layer=eval_layer, predict_layer=predict_layer)

    def _create_predict_fn(self, x_batch, predict_layer):
        mask_batch = T.matrix(dtype='int8')

        return theano.function(
            [x_batch, mask_batch],
            lasagne.layers.get_output(
                predict_layer,
                inputs=x_batch,
                deterministic=True,
                mask=mask_batch),
            on_unused_input='warn')


def inproduct_sigmoid_distance(target_embeddings, output):
    assert target_embeddings.ndim == output.ndim

    activation = T.nnet.sigmoid(
        T.sum(target_embeddings * output,
              axis=target_embeddings.ndim - 1))

    activation = T.clip(activation, 1e-7, 1.0 - 1e-7)

    return activation


class VectorSpaceLanguageModelBase(LanguageModelBase):

    def __init__(self,
                 batch_size, window_size,
                 num_negative_samples,
                 representations_init,
                 entity_representations_init,
                 regularization_lambda,
                 training_set,
                 validation_set):
        super(VectorSpaceLanguageModelBase, self).__init__(
            batch_size=batch_size,
            window_size=window_size,
            representations_init=representations_init,
            regularization_lambda=regularization_lambda,
            regularization_fn=l2_regularization,
            training_set=training_set, validation_set=validation_set,
            learning_method=lasagne.updates.adam)

        self.num_entities = entity_representations_init.shape[0]
        self.entity_representation_size = entity_representations_init.shape[1]

        clazz_distribution = np.ones(self.num_entities, dtype=np.float32)
        clazz_distribution /= clazz_distribution.sum()

        self.clazz_distribution = theano.shared(
            clazz_distribution, borrow=True)

        assert self.training_set[1].ndim == 1, \
            'Only one-hot vectors supported.'

        assert num_negative_samples is None or num_negative_samples >= 0, \
            'Number of negative samples should be None, zero or positive ' \
            '(currently: {0}).'.format(num_negative_samples)

        self.entity_representations = self._create_param(
            entity_representations_init, 'Class representations')

    def get_representations(self):
        return (self.projection_layer.get_representations(),
                self.entity_representations.get_value(borrow=True))

    def _negative_sampling(self, num_negative_samples, target_indices):
        assert num_negative_samples > 0

        logging.debug('Stochastically sampling %d negative instances '
                      'out of %d classes (%.2f%%).',
                      num_negative_samples, self.num_entities,
                      100.0 *
                      float(num_negative_samples) / self.num_entities)

        from theano.tensor.shared_randomstreams import RandomStreams

        srng = RandomStreams(
            seed=np.random.randint(low=0, high=(1 << 30)))

        rng_sample_size = (self.batch_size, num_negative_samples,)

        logging.debug(
            'Using %s for random sample generation of %s tensors.',
            RandomStreams, rng_sample_size)

        logging.debug('For every batch %d random integers are sampled.',
                      np.prod(rng_sample_size))

        random_negative_indices = srng.choice(
            rng_sample_size,
            a=self.num_entities,
            p=self.clazz_distribution)

        if self.__DEBUG__:
            random_negative_indices = theano.printing.Print(
                'random_negative_indices')(random_negative_indices)

        return random_negative_indices

    def _fetch_class_embeddings(self, indices):
        assert indices is not None

        # For the single positive example, indices is a vector of
        # just (batch_size,), when using negative sampling, it is a
        # two-dimensional tensor of size (batch_size, num_neg_samples).
        #
        # target_embeddings is a tensor of size
        # (batch_size, num_candidates, embedding_size)
        target_embeddings = T.take(
            self.entity_representations, indices, axis=0)

        if indices.ndim == 1:
            # This reshapes to (batch_size,
            #                   num_candidates=1,
            #                   embedding_size).
            target_embeddings = target_embeddings.dimshuffle(0, 'x', 1)

        assert target_embeddings.ndim == 3

        # At this stage, target_embeddings is a tensor of size
        # (batch_size, num_candidates, embedding_size) for all cases.
        target_embeddings = target_embeddings.dimshuffle(0, 'x', 1, 2)

        # Now, target_embeddings is of size
        # (batch_size, window_size=1, num_candidates, embedding_size).
        assert target_embeddings.ndim == 4

        return target_embeddings

    def _create_predict_fn(self, x_batch, predict_layer):
        mask_batch = T.matrix(dtype='int8')

        return theano.function(
            [x_batch, mask_batch],
            lasagne.layers.get_output(
                predict_layer,
                inputs=x_batch,
                deterministic=True,
                mask=mask_batch),
            on_unused_input='warn')


class VectorSpaceLanguageModel(VectorSpaceLanguageModelBase):

    def __init__(self,
                 batch_size, window_size,
                 num_negative_samples,
                 representations_init,
                 entity_representations_init,
                 regularization_lambda,
                 training_set,
                 validation_set):
        super(VectorSpaceLanguageModel, self).__init__(
            batch_size=batch_size,
            window_size=window_size,
            num_negative_samples=num_negative_samples,
            representations_init=representations_init,
            entity_representations_init=entity_representations_init,
            regularization_lambda=regularization_lambda,
            training_set=training_set,
            validation_set=validation_set)

        input_layer = lasagne.layers.InputLayer(
            shape=(self.batch_size, self.window_size))

        projection_layer = self._create_projection_layer(input_layer)
        previous_layer = projection_layer

        # Shape becomes (batch_size, embedding_size).
        previous_layer = MeanLayer(previous_layer, axis=1)

        self.transform_layers = []

        dense_nonlinearity = lasagne.nonlinearities.tanh

        transform = lasagne.layers.DenseLayer(
            previous_layer,
            num_units=self.entity_representation_size,
            nonlinearity=dense_nonlinearity,
            name='WordProjection')

        self.transform_layers.append(transform)

        previous_layer = ClipLayer(
            transform,
            lower_bound=-1.0 + 1e-7,
            upper_bound=1.0 - 1e-7)

        predict_layer = previous_layer

        def loss_fn(output, target_indices, input):
            # Perform negative sampling.
            random_negative_indices = self._negative_sampling(
                num_negative_samples, target_indices)

            word_projections = output.dimshuffle(0, 'x', 'x', 1)

            target_similarity = inproduct_sigmoid_distance(
                self._fetch_class_embeddings(target_indices),
                word_projections)\
                .dimshuffle(0)

            assert target_similarity.ndim == 1

            negative_samples_similarity = inproduct_sigmoid_distance(
                self._fetch_class_embeddings(random_negative_indices),
                word_projections).dimshuffle(0, 2)
            assert negative_samples_similarity.ndim == 2

            log_target_candidate = T.log(target_similarity)
            log_negative_candidate = \
                T.log(1.0 - negative_samples_similarity)

            objective = log_target_candidate + \
                T.sum(log_negative_candidate, axis=1, keepdims=False)

            return -objective

        self._finalize(loss_fn,
                       weight_layers=self.transform_layers,
                       projection_layers=[projection_layer,
                                          self.entity_representations],
                       eval_layer=predict_layer, predict_layer=predict_layer,
                       additional_params=[self.entity_representations])

    def _create_predict_fn(self, x_batch, predict_layer):
        avg_word_embedding = T.vector(dtype=theano.config.floatX)

        result = avg_word_embedding

        for transform in self.transform_layers:
            result = transform.get_output_for(
                result, deterministic=True)

        return theano.function(
            [avg_word_embedding], result,
            on_unused_input='warn')
