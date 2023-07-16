import random
import unittest

import curriculum.manual_curriculum
from curriculum.task_encoder import TaskEncoder

LLM_CHECKPOINT = "Salesforce/codegen-350M-mono"

# NOTE: models that are not Salesforce/codegen-350M-mono may give different number
EMBEDDING_DIM = 1024


class TestTaskEncoder(unittest.TestCase):
  # pylint: disable=protected-access,bad-builtin
  @classmethod
  def setUpClass(cls):
    cls.task_encoder = TaskEncoder(
        LLM_CHECKPOINT, curriculum.manual_curriculum, batch_size=1
    )  # see test_batch_process()

  def test_embed_dim(self):
    self.assertEqual(self.task_encoder.embed_dim, EMBEDDING_DIM)

  def test_task_encoder_api(self):
    task_spec = random.sample(curriculum.manual_curriculum.task_spec, 10)
    task_spec_with_embedding = self.task_encoder.get_task_embedding(task_spec)

    # TODO: automatically get the embedding dimension from the model info
    for single_spec in task_spec_with_embedding:
      self.assertFalse(sum(single_spec.embedding) == 0)

  def test_get_task_deps_src(self):
    custom_fn = curriculum.manual_curriculum.PracticeInventoryManagement
    fn_src, deps_src = self.task_encoder._get_task_deps_src(custom_fn)

    self.assertEqual(
        fn_src,
        "def PracticeInventoryManagement(gs, subject, space, num_tick):\n  "
        +
        "return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)\n",
    )
    self.assertTrue("def InventorySpaceGE(" in deps_src)
    self.assertTrue("def TickGE(" in deps_src)

  def test_contruct_prompt(self):
    single_spec = random.choice(curriculum.manual_curriculum.task_spec)
    prompt = self.task_encoder._construct_prompt(
        single_spec.reward_to, single_spec.eval_fn, single_spec.eval_fn_kwargs
    )
    print(prompt)

  def test_batch_process(self):
    batch_size = 8
    task_encoder = TaskEncoder(
        LLM_CHECKPOINT, curriculum.manual_curriculum, batch_size=batch_size
    )
    batch_embedding = task_encoder._get_embedding(
        ["# just to get the embedding size"] * batch_size
    )
    self.assertEqual(len(batch_embedding), batch_size)


if __name__ == "__main__":
  unittest.main()