# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators import cross_entropy


_test_value_type_integer_tensor = (tf.int32, (3,))
_test_value_type_float_tensor = (tf.float32, (3,))
_test_value_type_list_integer_tensors = [(tf.int32, (2,)),
                                         (tf.int32, (3,))]


class CrossEntropyComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer_tensor', _test_value_type_integer_tensor))
  def test_cross_entropy_properties(self, value_type):
    factory = cross_entropy.CrossEntropyFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.StructType([
        ('cross_entropy_gamma', tf.float32),
        ('cross_entropy_delta', tf.float32)
    ])
    expected_measurements_type = tff.type_at_server(expected_measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=expected_measurements_type))
    tff.test.assert_types_equivalent(process.next.type_signature,
                                     expected_next_type)

  @parameterized.named_parameters(
      ('float_tensor', _test_value_type_float_tensor),
      ('list_integer_tensors', _test_value_type_list_integer_tensors))
  def test_cross_entropy_create_raises(self, value_type):
    factory = cross_entropy.CrossEntropyFactory()
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class CrossEntropyExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('integer_tensor', _test_value_type_integer_tensor))
  def test_cross_entropy_impl(self, value_type):
    factory = cross_entropy.CrossEntropyFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [range(1, value_type.shape[0] + 1) for _ in range(2)]
    expected_result = [x * 2 for x in range(1, value_type.shape[0] + 1)]
    expected_measurements = collections.OrderedDict(
        cross_entropy_gamma=2.3333333333,
        cross_entropy_delta=3.)

    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ('integer_tensor', _test_value_type_integer_tensor))
  def test_cross_entropy_zeros_impl(self, value_type):
    factory = cross_entropy.CrossEntropyFactory()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [range(0, value_type.shape[0]) for _ in range(2)]
    expected_result = [x * 2 for x in range(0, value_type.shape[0])]
    expected_measurements = collections.OrderedDict(
        cross_entropy_gamma=4./3.,
        cross_entropy_delta=5./3.)

    measurements = process.next(state, client_values).measurements
    self.assertAllClose(measurements, expected_measurements)
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

if __name__ == '__main__':
  tf.test.main()
