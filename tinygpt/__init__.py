"""tiny-GPT-trainer: the core trainer.

Everything that defines, trains, or samples from the model lives under this
package: the tokenizer, the data adapters, the token-shard preparation, the
transformer itself, the training loop, and the sampler.

It is deliberately separate from `web/`, which is an optional viewer that
shells out to the entry points here and parses their output. The dependency
runs one way only -- nothing in `tinygpt` imports anything from `web` -- so
the trainer stays a standalone command-line project that happens to have a
dashboard available, not one that needs it.

Every module here is runnable on its own, as `python -m tinygpt.<module>`:

    python -m tinygpt.config                    # preset + parameter breakdown
    python -m tinygpt.model                     # architecture smoke demos
    python -m tinygpt.tokenizer.tokenizer       # BPE encode/decode round trip
    python -m tinygpt.adapters.plain_text       # adapter chunking demo
    python -m tinygpt.sample                    # sampling-knob demo, no model
"""
