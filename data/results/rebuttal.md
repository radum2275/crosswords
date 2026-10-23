Rebuttal:
We sincerely thank the reviewer for their insightful and constructive feedback. Below, we address your comments and questions with detailed responses.

W1:

In Romanian crosswords polysemantic puzzles are clearly separated from thematic puzzles. Virtually all clues in polysemantic clues are polysemantic, and virtually all clues thematic puzzles are not polysemantic (i.e., they have a straight meaning). Puzzles are validated by the editors before publication, and this includes the validation of clues as being genuinely polysemantic in polysemantic puzzles.

Regarding the creation of PolyCross (also response to Q1): We created the PolyCross dataset from polysemantic puzzles available on the website Rebus Online, rebusonline.com. All polysemantic puzzles considered have a grid size no greater than 12x12, which further implies that all solution words in PolyCross have a length no greater than 12. Given all solution words in PolyCross, we exclude from our analysis two-letter solutions, and keep all remaining words within the PolyCross. Skipping two-letter solutions has the following motivation, also summarized in the paper. Any combination of two letters is acceptable in a grid, which implies that often these are not actual words in the Romanian language. This is why they often require a special technique to construct the clues, which is beyond the scope of our paper. (Here is an example: SC = School entrance! Note how this is a domain specific play on letters, which once again is beyond the scope of our paper).

W2: We agree that the difficulty to translate polysemantic expressions is not a surprise. We believe, however, that it is worthwhile to point it out, and to quantify this difficulty with an experiment, given that translating plain short expressions is much easier.

W3: Please see the response to Q2.

W4: Here is the equivalent of the paragraph from line 200 to 207 applied to ThemCross:

“As the prefix length increases accuracy improves as more letters from the solution are given. With the longest prefix of four letters in the hint the accuracy reaches 84.9% for gpt-oss-120b on ThemCross. (The newly evaluated Claude Opus 4.8 reaches an accuracy of 95.4% with the longest, four-letter hints on ThemCross, and 85.7% on PolyCross, as seen in the table line provided in our rebuttal). To better understand the contribution of the hint (word length and prefix), note that 44.68% of all solutions in ThemCross have no more than four letters. Thus the hint with four letters has enough information to uniquely identify the solution in all those cases. An additional 22.29% of the solutions have a length of 5 in which case the prefix gives 80% of the correct letters. 17.68% of the solutions have a length of 6. All these cases with words of up to 6 letters, where the hint with four letters give two thirds of the required letters or more, add up to 84.66% of all solutions.”

Consider the case where the hint includes the solution length plus the first 4 letters (or fewer if the word is shorter), and consider all instances where the solution length is 4 or shorter. In these instances the hint is sufficient to give the solution even without analysing the clue. In these instances, gpt-oss-120b has an accuracy of 98.7% on PolyCross and 98.67% on ThemCross.

Q1: Please see the response to W1.

Q2: Unfortunately, we did not have access to Romanian-trained LLMs. However we evaluated a frontier model, Claude Opus 4.8, which is widely recognized for its strong multilingual capabilities. The results were broadly consistent with those obtained using the open-source models, particularly on the more challenging PolyCross dataset. Specifically, when using the No Hints (v1) and Answer Length (v2) prompting strategies, Claude Opus 4.8 achieved accuracies of 0.052 and 0.086, respectively. The table below summarizes the results across all prompt variants.

model	v1	v2	v3	v3	v3	v3	v4	v4	v4	v4	v5
prefix len				1	2	3	4	1	2	3	4
granite-4.0-small	0.006	0.010	0.042	0.116	0.280	0.396	0.022	0.050	0.160	0.253	0.005
llama-3.3-70b	0.016	0.034	0.111	0.243	0.433	0.523	0.063	0.106	0.379	0.503	0.036
gpt-oss-120b	0.027	0.075	0.202	0.347	0.576	0.727	0.153	0.277	0.464	0.585	0.058
claude-opus 4.8	0.052	0.086	0.284	0.507	0.758	0.857	0.296	0.507	0.759	0.860	0.087
Q3: We experimented with several variations of the prompts described in the appendix. However, our preliminary results showed only minor differences across prompt formulations, with no consistent improvements in performance. This suggests that the task difficulty is driven primarily by the underlying reasoning challenges rather than by prompt engineering choices. We will nevertheless continue to explore alternative prompting strategies in future work.

In all our experiments, the temperature was set to 0.

Thank you as well for pointing out the small typos which we will fix.