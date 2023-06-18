# Are Hate Speech classifiers biased against AAE? A counterfactual probing investigation

There is [extensive evidence](https://aclanthology.org/P19-1163/) that hate speech detection models are biased against African American English (AAE, as opposed to Standard American English (SAE)). However, do the models actually *use* dialect information when making their decision? 

Inspired by work in [Counterfactual probing](https://aclanthology.org/2021.conll-1.15/), we wanted to investigate if by learning decision boundaries corresponding to dialect (SAE vs AAE) using AlterRep, we can test how hate speech models respond to controlled interventions that change only the dialect of an utterance, at the representation level.

## Experiments

For data, we use the intent equivalent AAE-SAE pairs from [Groenwold et. al 2020](https://github.com/sophiegroenwold/AAVE_SAE_dataset). We train [INLP](https://aclanthology.org/2020.acl-main.647/) classifiers on these pairs of utterances, thus giving us a dialect boundary in model space.

We use the [`cardiffnlp/twitter-roberta-base-offensive`](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive) model as a hate speech detector, as the model was finetuned on Twitter data (which overlaps with the utterance types in the dataset we use to train linear classifiers), as well as for ease of use.

All code used to produce results is present in this repository.

## Results

The following bar charts show the proportion of the dataset that was classified as hate speech after intervention, with the transformer layer where the intervention was performed on the X axis. The first bar chart is for the AAE intervention (that is, pushing representations to be more towards the AAE side of the learned linear decision boundary), while the second bar chart is for the SAE intervention.

![Bar chart showing the proportion of the dataset classified as hate speech after intervention on y axis, transformer layer where intervention was performed on x axis, for the intervention that makes model representations 'more AAE'](https://github.com/venkatasg/aae-probing/aae.png?raw=true)

![Bar chart showing the proportion of the dataset classified as hate speech after intervention on y axis, transformer layer where intervention was performed on x axis, for the intervention that makes model representations 'more SAE'](https://github.com/venkatasg/aae-probing/sae.png?raw=true)



## Conclusion

There seems to be some support that pushing model representations to be more AAE leads to a higher probability for the model to classify utterances as hate speech, as the figures above seem to show.

However, these results are not conclusive, as the variability in the same plots show. Moreover, further experiments need to incorporate more diverse datasets like [VALUE](https://aclanthology.org/2022.acl-long.258/), as well as more naturalistic examples of AAE.

## End

This work was a joint project between [Gauri Kambhatla](https://gaurikambhatla.github.io) and me (Venkata S Govindarajan) as part of a course project under [Kyle Mahowald](https://mahowak.github.io).

You can cite this work by linking to this GitHub repository.
