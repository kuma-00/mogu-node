import test from 'ava'

import { MoguImageProcessor, MoguFoodDetector, MoguTensor } from '../index'

test('MoguImageProcessor is exported', (t) => {
  t.truthy(MoguImageProcessor)
  t.is(typeof MoguImageProcessor, 'function')
})

test('MoguFoodDetector is exported', (t) => {
  t.truthy(MoguFoodDetector)
  t.is(typeof MoguFoodDetector, 'function')
})

test('MoguTensor is exported', (t) => {
  t.truthy(MoguTensor)
  t.is(typeof MoguTensor, 'function')
})

test('MoguImageProcessor can be instantiated', (t) => {
  const processor = new MoguImageProcessor(224, 224)
  t.truthy(processor)
})

test('MoguImageProcessor.withImagenetNormalization factory method works', (t) => {
  const processor = MoguImageProcessor.withImagenetNormalization(224, 224)
  t.truthy(processor)
})
