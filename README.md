
## related work explain
需为每个视觉元素训练单独的线性 SVM 检测器。
要对数据集中每幅图像的所有视觉元素与整个数据集进行两两比较来定位最近邻和构建聚类，其计算复杂度会随数据集规模呈二次方增长。 
计算成本过高 
具有更好的扩展性。

A separate linear SVM detector needs to be trained for each visual element.

To locate the nearest neighbors and construct clusters, pairwise comparisons must be made between all the visual elements of each image in the dataset and the entire dataset. The computational complexity grows quadratically with the size of the dataset.

The computational cost is prohibitively high.

It has better scalability. 

## expl for why use difffusion model
（利用大模型所学到的知识）
扩散模型的权重中隐式地存储输入数据的视觉特征。我们想利用这里的信息帮助数据挖掘。

(Utilizing the knowledge learned by the large model)
The visual features of the input data are implicitly stored in the weights of the diffusion model. We want to utilize this information to assist in data mining.

if you yolo you can not recognize these separate thing like spire red roof.

## CONTENT

 By fine-tuning conditional diffusion models on specific datasets and defining a typicality measure, it can effectively mine visual patterns in datasets with different tags such as time, geography, and semantic labels.
The method overcomes the scalability issues faced by traditional visual data mining techniques. Unlike traditional methods that require pairwise comparisons of all visual elements in the dataset, this approach does not rely on such comparisons and can handle large and diverse datasets, including historical car datasets, face datasets, street-view datasets, and scene datasets.
The paper also explores two applications. One is to analyze the trends of visual elements across different locations by translating geographical elements and mining typical translations. The other is to apply the typicality measure to medical images and improve the localization of abnormal areas in X-ray images of patients with various thorax diseases.

quadratic computational complexity
can not deploy on big dataset.

Scalability: Traditional SVM-based visual data mining methods require pairwise comparisons of all visual elements in the dataset to locate nearest neighbors and establish clusters. This leads to a quadratic increase in computational complexity with the growth of the dataset size. In contrast, the method proposed in this paper does not rely on such pairwise comparisons. It uses a diffusion model-based analysis-by-synthesis approach, which can scale well to very large datasets. For example, it can handle datasets like G^3 with 344K images and Places with 1.8M images, while SVM methods would struggle with such large-scale data.

svm can only identify the most typical image region
 iconic elements like aviator glasses in the 1920s and military hats in the 1940s,
svm can not identify so exactly
Ability to Mine Typical Elements: SVM-based works mainly focus on discriminative tasks rather than actually mining typical visual elements in the dataset. This paper, however, is designed specifically to identify the most typical image regions. By defining a typicality measure based on the performance of the diffusion model in denoising images under different label conditions, it can effectively extract characteristic visual patterns. For instance, in the face dataset, it can discover iconic elements like aviator glasses in the 1920s and military hats in the 1940s, which SVM methods are not capable of.


# Refined Speech for Presentation

Hello everyone,

Thank you for joining me today. I’m excited to present  the paper, **"Diffusion Models as Data Mining Tools,"** published at ECCV 2024 spot. 
Let’s explore together.
---

To start, let me pose a question: What makes Prague look like Prague?
 When we think about the essence of a city, 
** certain elements stand out.** For Prague, these could be:

1. **The Clock Tower** – A symbol of the city’s **rich historical tapestry.**
2. **The Spires** – Towering and sacred, embodying European heritage.
3. **The Red Roofs** – since i come from asia, i find it very eye-catching just like what i think the fairy tale town is.

While humans intuitively identify these features, achieving this computationally is challenging. This paper addresses the challenge using **diffusion models** as tools for visual data mining.

---

#### Diffusion Models Overview

Diffusion models work through two main processes:

1. **Forward Process**: Incrementally adds noise to data.
2. **Backward Process**: Removes noise to reconstruct the data with a trained model.

This iterative approach helps extract meaningful patterns. The model discussed here, **Stable Diffusion**, introduces two key innovations:

1. **Conditional Diffusion**: Guides generation using prompts, such as text labels.
2. **Latent Diffusion**: Operates in a compressed latent space, improving efficiency.

---

#### Typicality in Visual Data Mining

The concept of "typicality" is central. A visual element is considered typical if the model denoises data more effectively with a label. For instance:

- Without the label “Prague,” the model struggles.
- With the label, the loss decreases, indicating the element’s relevance.

By calculating typicality scores, we can pinpoint the most characteristic features in a dataset.

---
Details of Mining
Conditioning and Fine - Tuning: In the first street - view data, the new conditioning changes from an empty stream to a Google street - view image. For fine - tuning, we train each model on the target dataset by optimizing the loss.
Clustering: We collect patches with high typicality, say 1000 out of 10,000 pictures. Then we embed these patches in DFT features, considering the time parameter. After getting the features, we run k - means on them. The DFT (Diffusion Features) for an image  involves adding noise at time step , feeding it to the diffusion model, and extracting intermediate - layer activations.
Datasets and Results
The paper uses four datasets: cars, faces, geometry, and places. The first two are small, and the latter two are large and newly used in this paper. The results show that for a label like "soccer field," we can distinguish typical and non - typical pictures. Fine - tuning also helps in highlighting more typical elements

#### Applications


**Styles of Different Countries**: Instead of fixing the label \(C\), we change it, for example, from a Japanese photo to a UK photo (\(C_1\)). We generate a new picture \(XC_1\) and calculate the typicality. By selecting 10 countries in the dataset, we can find trends. For instance, roof pictures often have high typicality, and roof colors can vary between countries.

**Medical Images**: The paper uses a dataset of 8000 medical images with 7 disease classes, marked with red rectangles indicating regions of interest. Our fine - tuned model shows good results in finding the ROI for tuberculosis. The pre - trained model doesn't perform as well as the fine - tuned one.

---

#### Limitations

1. **Clustering Challenges**: Unrelated elements, like eyes and mouths, may cluster together.
2. **Noise Artifacts**: Noise introduced during feature extraction can reduce accuracy.
3. **Limited Generalization**: Fine-tuned models may overfit specific datasets.

---

#### Conclusion

In conclusion, **diffusion models** are powerful tools for visual data mining, uncovering patterns in cultural and medical contexts. They bridge generative AI with practical applications, offering a glimpse into future innovations.

Thank you for your attention. I look forward to your feedback and any questions you might have!



解说词

2

Clock Tower
The architectural style of the tower has a historical feel,

Spire
style appears magnificent

(Red Roof)
since i come from asia, i find it very eye-catching just like what i think the fairy tale town is.

![alt text](image.png)

我们的目标是提供不同标签的典型元素的可视化总结，例如使我们能够确定街
景全景位置的共同元素。

应用1
(c) Translation (Sec. 5.1) of a picture of a road
from France (top) to Thailand without finetuning (middle) suffers from data biases in
the base model turning the road into a river and erasing utility poles. 

CLIP feature obtained through contrastive learning

We translate 1000 images for
each of the 10 selected countries to all others, resulting in 100K images,

japan 1000 generate 10000 pic
all 10 generate 100000 pic


technical issues in making photos.

17
they all contains roofs.
means roofs pic are all typical 

but like in japan roofs are most in black color.
show the trend in roof pics.

# GPT4 paper assistant: A daily ArXiv scanner

This repo implements a very simple daily scanner for Arxiv that uses GPT4 and author matches to find papers you might find interesting. 
It will run daily via github actions and can post this information to slack via a bot or just render it in a static github-pages website.

A simple demo of the daily papers can be seen [here](https://tatsu-lab.github.io/gpt_paper_assistant/) running on `cs.CL`

As a cost estimate, running this on all of `cs.CL` cost $0.07 on 2/7/2024

## Changelog
- **2/15/2024**: fixed a bug with author parsing in the RSS format + cost estimates for title filtering being off + crash when 0 papers are on the feed. 
- **2/7/2024**: fixed a critical issue from ArXiv changing their RSS format. Added and enabled a title filtering to reduce costs.


## Quickstart
This is the minimal necessary steps to get the scanner to run. It is highly recommended to read the whole thing to decide what you want to run.

### Running on github actions

1. Copy/fork this repo to a new github repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow) if you fork it.
2. Copy `config/paper_topics.template.txt` to `config/paper_topics.txt` and fill it out with the types of papers you want to follow
3. Copy `config/authors.template.txt` to `config/authors.txt` and list the authors you actually want to follow. The numbers behind the author are important. They are semantic scholar author IDs which you can find by looking up the authors on semantic scholar and taking the numbers at the end of the URL.
4. Set your desired ArXiv categories in `config/config.ini`.
5. Set your openai key (`OAI_KEY`) as ``a [github secret](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository)
6. In your repo settings, set github page build sources to be [github actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow)

At this point your bot should run daily and publish a static website. You can test this by running the github action workflow manually.

**Optional but highly recommended**: 

7. Get and set up a semantic scholar API key (`S2_KEY`) as a github secret. Otherwise the author search step will be very slow
8. [Set up a slack bot](https://api.slack.com/start/quickstart), get the OAuth key, set it to `SLACK_KEY` as a github secret
9. Make a channel for the bot (and invite it to the channel), get its [Slack channel id](https://stackoverflow.com/questions/40940327/what-is-the-simplest-way-to-find-a-slack-team-id-and-a-channel-id), set it as `SLACK_CHANNEL_ID` in a github secret.
10. Take a look at `configs/config.ini` to tweak how things are filtered.
11. Set the github repo private to avoid github actions being [set to inactive after 60 days](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow)

Each day at 1pm UTC, the bot will run and post to slack and publish a github pages website (see the publish_md and cron_runs actions for details).

### Running locally

The steps are generally the same as above, but you have to set up the environment via `requirements.txt`

Instead of passing credentials via github secrets, you have to set environment variables `OAI_KEY`, `SLACK_KEY`, `SLACK_CHANNEL_ID`.

To run everything, just call `main.py`

**Other notes:**
You may also want to not push to slack, in which case set your desired output endpoint (json, markdown, slack) in the `dump_json`, `dump_md`, and `push_to_slack` fields of `config/config.ini`.

If the semantic scholar API times out or is slow, you should get a [S2 api key](https://www.semanticscholar.org/product/api#api-key-form) and set it as `S2_KEY` in your environment variables.
(due to the limitations of github actions, this will only help if the code is run locally)

**Making it run on its own:**
This whole thing takes almost no compute, so you can rent the cheapest VM from AWS, put this repo in it, install the `requirements.txt`
appropriately set up the environment variables and add the following crontab
```
0 13 * * * python ~/arxiv_scanner/main.py
```
This crontab will run the script every 1pm UTC, 6pm pacific. 

## Making the `paper_topics.txt` prompt
The `paper_topics.txt` file is used to generate the prompt for GPT. It is a list of topics that you want to follow.
One set of examples might be something like 
```text
 1. New methodological improvements to RLHF or instruction-following which are specific fine-tuning steps that are taken to make language models better at following user instructions across a range of tasks.
    - Relevant: papers that discuss specific methods like RLHF, or instruction-tuning datasets, improving these methods, or analyzing them.
    - Not relevant: papers about adaptation to some task. Simply following instructions or inputs are not sufficient.
 2. Shows new powerful test set contamination or membership inference methods for language models. Test set contamination is the phenomenon where a language model observes a benchmark dataset during pretraining.
    - Relevant: test statistics that can detect contamination of benchmarks in language models. statistics that can provide guarantees are more interesting. membership inference methods that are general enough to apply to language models are also relevant.
    - Not relevant: any papers that do not consider language models, or that do not consider test set contamination.
 3. Shows a significant advance in the performance of diffusion language models.
    - Relevant: papers that study language models that are also diffusion models. Continuous diffusions are even more relevant, while discrete diffusions are less so.
    - Not relevant: papers about image diffusions like DALL-E or Stable Diffusion, or papers that do not explicitly mention language models or applications to text.
```
This is just a standard prompt, but being very specific can help, especially for things like 'diffusion language models' or 'instruction-following', where the LM can get confused about whether image diffusions are relevant, or if doing some task better is sufficient to improve instruction following.

You may also want to follow this with some general interest areas like
```text
In suggesting papers to your friend, remember that he enjoys papers on statistical machine learning, and generative modeling in natural language processing.
 Your friend also likes learning about surprising empirical results in language models, as well as clever statistical tricks.
 He does not want to read papers that are about primarily applications of methods to specific domains.
```

## Details of how it works

The script grabs a candidate set of ArXiv papers for a specific day, via the RSS feeds. To avoid double-announcing papers, it will only grab an RSS feed within the last day. To avoid missing papers, you'd want to run this every day. 
It filters out any `UPDATED` papers and announces only new ones.

The filtering logic is pretty simple. We first check for author match.
1. Do a lookup of the authors on semantic scholar, getting a list of candidate matches.
2. Check the authors of the paper. If the author semantic scholar id matches someone in `authors.txt` it goes in the candidate set with a default score of `author_match_score`.

We then check for GPT-evaluated relevance. We do this in two steps.
1. Filter out any papers that have no authors with h-index above `hcutoff` in `config.ini`. This is to reduce costs.
2. All remaining examples get batched, and are evaluated by a GPT model specified by `model` in `config.ini`. **You should only use GPT3.5 for debugging. It does not work well for this purpose!**
This step uses the following prompt setup defined in `configs/`

>You are a helpful paper reading assistant whose job is to read daily posts from ArXiv and identify a few papers that might be relevant for your friend. There will be up to 5 papers below. Your job is to find papers that:
> 1. Criterion 1
> 2. Criterion 2
> 
> [PAPERS]
> 
> Write the response in JSONL format with {ARXIVID, COMMENT, RELEVANCE, NOVELTY} on each line, one for each paper.
The ARXIVID should be the ArXiv ID.
The COMMENT should identify whether there is a criteria that match the paper very closely. If so, it should mention it by number (no need to mention the non-matching criteria).
These matches should not be based on general terms like "language modeling" or "advancements" and should specifically refer to a criterion.
The RELEVANCE should be a relevance score from 1-10 where 10 must be directly related to the exact, specific criterion with near-synonym keyword matches and authors who are known for working on the topic, 1 is irrelevant to any criterion, and unrelated to your friend's general interest area, 2-3 is papers that are relevant to the general interest area, but not specific criteria, and 5 is a direct match to a specific criterion.
The NOVELTY should be a score from 1 to 10, where 10 is a groundbreaking, general-purpose discovery that would transform the entire field and 1 is work that improves one aspect of a problem or is an application to a very specific field. Read the abstract carefully to determine this and assume that authors cannot be trusted in their claims of novelty.

3. GPT scores the papers for relevance (to the topics in `config/papers_topics.txt`) and novelty (scale 1-10)
4. Papers are filtered if they have scores below either the relevance and novelty cutoffs in `config.ini`
5. Papers are given an overall score based on equal weight to relevance and novelty

Finally, all papers are sorted by the max of their `author_match_score` and the sum of the GPT-rated relevance and novelty scores (the relevance and novelty scores will only show up in the final output if they are above the cutoff thresholds you set in the config file). Then the papers are rendered and pushed into their endpoints (text files or Slack).

## Contributing 
This repo uses ruff - `ruff check .` and `ruff format .` 
Please install the pre-commit hook by running `pre-commit install`

### Testing and improving the GPT filter
The `filter_papers.py` code can also be run as a standalone script.
This will take a batch of papers in `in/debug_papers.json`, run whatever config and prompts you have
and return an output to `out/filter_paper_test.debug.json`. If you find the bot makes mistakes, you can find the associated batch in `out/gpt_paper_batches.debug.json` and copy that into the relevant `debug_papers` file.

This lets you build a benchmark for the filter and to see what comes out on the other side.

## Other stuff
This repo and code was originally built by Tatsunori Hashimoto is licensed under the Apache 2.0 license.
Thanks to Chenglei Si for testing and benchmarking the GPT filter.
