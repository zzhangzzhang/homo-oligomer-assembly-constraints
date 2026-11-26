# Homo-oligomeric contacts and protein assembly

Code and processed data supporting **Protein language model scaling and AI literature mining uncover subfamily-specific assembly constraints and overlooked biological interfaces**.

This repository examines how protein models recover contacts between identical subunits and combines coevolutionary analysis with literature evidence to assess PDB biological assemblies.

## Workflows

| Workflow | Contents |
| --- | --- |
| [Core analyses](workflows/core_figures/README.md) | Protein language model contact prediction, model scaling, MSA selection, and AlphaFold2-Multimer query-conditioning and covariance-ablation experiments. |
| [Literature and PDB assemblies](workflows/literature_pdb/README.md) | LLM-assisted extraction of oligomerization evidence, MSA Pairformer contact analysis, PISA interface assessment, and integration with deposited assemblies. |
| [Alternative conformations](workflows/af2_alternative_states/README.md) | Selective covariance ablation in AlphaFold2 to shift predictions between experimentally observed conformations. |

Each workflow README describes the scripts, execution order, inputs, outputs, and dependencies.


## License

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licensing and third-party notices.
