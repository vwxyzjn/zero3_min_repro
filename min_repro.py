import torch
import transformers
import torch.nn.functional as F
import accelerate
from accelerate.state import AcceleratorState
from datasets import load_dataset
import datasets
import argparse
import deepspeed

def prepare_deepspeed2(args, ref_policy):
    deepspeed_states = AcceleratorState().deepspeed_plugin
    deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    eval_ds_config = {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
        "bf16": {"enabled": True},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
    ref_policy.eval()
    print("🔥 deepspeed2 is initialized")
    return ref_policy


def prepare_deepspeed3(args, model, accelerator):
    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    # deepspeed_states = AcceleratorState().deepspeed_plugin
    # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    print("🔥 deepspeed3 is initialized")
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--deepspeed2", action="store_true")
parser.add_argument("--deepspeed3", action="store_true")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--micro_batch_size", type=int, default=1)
args = parser.parse_args()


dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144", split="train")
dataset = dataset.with_format("torch", columns=["query_token", "reference_response_token"])
dummy_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

accelerator = accelerate.Accelerator()
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", padding_side="right")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
accelerator.print("=================")
policy = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
ref_policy = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
for model in [policy, ref_policy]:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

dummy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
policy, dummy_dataloader, dummy_optimizer = accelerator.prepare(policy, dummy_dataloader, dummy_optimizer)
accelerator.print({
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "deepspeed": deepspeed.__version__,
    "datasets": datasets.__version__,
})
if args.deepspeed2:
    ref_policy = prepare_deepspeed2(args, ref_policy)
elif args.deepspeed3:
    ref_policy = prepare_deepspeed3(args, ref_policy, accelerator)
else:
    ref_policy = ref_policy.to(accelerator.device)



for data in dummy_dataloader:
    query = data["query_token"]
    break

temperature = 0.7
context_length = query.shape[1]
def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # already handled in generation
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_logits=True
    )
    logits = torch.stack(output.logits, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits

generation_config = transformers.GenerationConfig(
    max_new_tokens=50,
    min_new_tokens=50,
    temperature=temperature,
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
)
# rollout 
with torch.no_grad():
    query_response, logits = generate(accelerator.unwrap_model(policy), query, tokenizer, generation_config)
    response = query_response[:, context_length:]
    logits /= temperature + 1e-7
    all_logprob = F.log_softmax(logits, dim=-1)
    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
    # accelerator.print(f"{response=}")
    # accelerator.print(f"{logprob=}")

    ref_output = forward(ref_policy, query_response, tokenizer)
    ref_logits = ref_output.logits[:, context_length - 1 : -1]
    ref_logits /= temperature + 1e-7
    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
    # accelerator.print(f"{ref_logprob=}")
    accelerator.print(f"{(ref_logprob-logprob).exp().mean()=}")
    torch.testing.assert_close(ref_logprob, logprob, rtol=1e-2, atol=1e-2) # a very generous tolerance
