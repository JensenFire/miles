from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    train_tp: int = 1
    train_ep: int = 1
    train_pp: int = 1
    sglang_tp: int = 1
    sglang_dp: int = 1
    sglang_ep: int = 1
    sglang_pp: int = 1

    num_train_gpus: int = 1
    num_rollout_gpus: int = 1

    def validate(self):
        assert self.sglang_pp == 1, "Not supported yet for sglang pp"
        assert (
            self.num_train_gpus % (self.train_pp * self.train_tp) == 0
        ), "num_train_gpus must be divisible by train_tp and pp"
        assert (
            self.num_train_gpus % (self.train_pp * self.train_ep) == 0
        ), "num_train_gpus must be divisible by train_ep and pp"
        assert self.num_rollout_gpus + self.num_train_gpus <= 8, "Not enough GPUs available"


def prepare(args: ScriptArgs):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command("hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    num_gpus = args.num_train_gpus + args.num_rollout_gpus
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=num_gpus)


def execute(args: ScriptArgs):

    num_gpus = args.num_train_gpus + args.num_rollout_gpus
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        "--balance-data "
    )
    # Training parallellism settings
    perf_args = (
        f"--tensor-model-parallel-size {args.train_tp} "
        f"--pipeline-model-parallel-size {args.train_pp} "
        f"--expert-model-parallel-size {args.train_ep} "
        f"--expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.sglang_tp} "
        f"--rollout-num-gpus {args.num_rollout_gpus} "
        f"--sglang-data-parallel-size {args.sglang_dp} "
        f"--sglang-expert-parallel-size {args.sglang_ep} "
        f"--sglang-pipeline-parallel-size {args.sglang_pp} "
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_train_gpus} "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        "--check-weight-update-equal "
        "--update-weight-transfer-mode rdma "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    args.validate()
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
