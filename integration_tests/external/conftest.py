import pytest

from valor import Annotation, Datum, GroundTruth, Prediction


@pytest.fixture
def answer_correctness_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""Did John Adams get along with Alexander Hamilton?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def answer_correctness_q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Did Lincoln win the election of 1860?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def answer_correctness_datums(
    answer_correctness_q0: Datum,
    answer_correctness_q1: Datum,
) -> list[Datum]:
    return [answer_correctness_q0, answer_correctness_q1]


@pytest.fixture
def answer_correctness_predictions() -> list[str]:
    return [
        """John Adams and Alexander Hamilton did not get along. John Adams and Alexander Hamilton were both federalists.""",
        """Lincoln won the election of 1860.""",
    ]


@pytest.fixture
def answer_correctness_groundtruths() -> list[str]:
    return [
        """John Adams and Alexander Hamilton did not get along. John Adams and Alexander Hamilton held opposing views on the role of the federal government.""",
        """Lincoln won the election of 1860.""",
    ]


@pytest.fixture
def answer_correctness_gt_questions(
    answer_correctness_datums: list[Datum],
    answer_correctness_groundtruths: list[str],
) -> list[GroundTruth]:
    assert len(answer_correctness_datums) == len(
        answer_correctness_groundtruths
    )
    return [
        GroundTruth(
            datum=answer_correctness_datums[i],
            annotations=[Annotation(text=answer_correctness_groundtruths[i])],
        )
        for i in range(len(answer_correctness_datums))
    ]


@pytest.fixture
def answer_correctness_pred_answers(
    answer_correctness_datums: list[Datum],
    answer_correctness_predictions: list[str],
) -> list[GroundTruth]:
    assert len(answer_correctness_datums) == len(
        answer_correctness_predictions
    )
    return [
        Prediction(
            datum=answer_correctness_datums[i],
            annotations=[
                Annotation(
                    text=answer_correctness_predictions[i],
                )
            ],
        )
        for i in range(len(answer_correctness_datums))
    ]


@pytest.fixture
def answer_relevance_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""Did John Adams get along with Alexander Hamilton?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def answer_relevance_q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Did Lincoln win the election of 1860?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def answer_relevance_datums(
    answer_relevance_q0: Datum,
    answer_relevance_q1: Datum,
) -> list[Datum]:
    return [answer_relevance_q0, answer_relevance_q1]


@pytest.fixture
def answer_relevance_predictions() -> list[str]:
    return [
        """John Adams and Alexander Hamilton did not get along.""",
        """If a turtle egg was kept warm, it would likely hatch into a baby turtle. The sex of the baby turtle would be determined by the incubation temperature.""",
    ]


@pytest.fixture
def answer_relevance_context_list() -> list[list[str]]:
    return [
        [
            """Although aware of Hamilton\'s influence, Adams was convinced that their retention ensured a smoother succession. Adams maintained the economic programs of Hamilton, who regularly consulted with key cabinet members, especially the powerful Treasury Secretary, Oliver Wolcott Jr. Adams was in other respects quite independent of his cabinet, often making decisions despite opposition from it. Hamilton had grown accustomed to being regularly consulted by Washington. Shortly after Adams was inaugurated, Hamilton sent him a detailed letter with policy suggestions. Adams dismissively ignored it.\n\nFailed peace commission and XYZ affair\nHistorian Joseph Ellis writes that "[t]he Adams presidency was destined to be dominated by a single question of American policy to an extent seldom if ever encountered by any succeeding occupant of the office." That question was whether to make war with France or find peace. Britain and France were at war as a result of the French Revolution. Hamilton and the Federalists strongly favored the British monarchy against what they denounced as the political radicalism and anti-religious frenzy of the French Revolution. Jefferson and the Republicans, with their firm opposition to monarchy, strongly supported the French overthrowing their king. The French had supported Jefferson for president in 1796 and became belligerent at his loss.""",
            """Led by Revolutionary War veteran John Fries, rural German-speaking farmers protested what they saw as a threat to their liberties. They intimidated tax collectors, who often found themselves unable to go about their business. The disturbance was quickly ended with Hamilton leading the army to restore peace.Fries and two other leaders were arrested, found guilty of treason, and sentenced to hang. They appealed to Adams requesting a pardon. The cabinet unanimously advised Adams to refuse, but he instead granted the pardon, arguing the men had instigated a mere riot as opposed to a rebellion. In his pamphlet attacking Adams before the election, Hamilton wrote that \"it was impossible to commit a greater error.\"\n\nFederalist divisions and peace\nOn May 5, 1800, Adams's frustrations with the Hamilton wing of the party exploded during a meeting with McHenry, a Hamilton loyalist who was universally regarded, even by Hamilton, as an inept Secretary of War. Adams accused him of subservience to Hamilton and declared that he would rather serve as Jefferson's vice president or minister at The Hague than be beholden to Hamilton for the presidency. McHenry offered to resign at once, and Adams accepted. On May 10, he asked Pickering to resign.""",
            """Indeed, Adams did not consider himself a strong member of the Federalist Party. He had remarked that Hamilton\'s economic program, centered around banks, would "swindle" the poor and unleash the "gangrene of avarice." Desiring "a more pliant president than Adams," Hamilton maneuvered to tip the election to Pinckney. He coerced South Carolina Federalist electors, pledged to vote for "favorite son" Pinckney, to scatter their second votes among candidates other than Adams. Hamilton\'s scheme was undone when several New England state electors heard of it and agreed not to vote for Pinckney. Adams wrote shortly after the election that Hamilton was a "proud Spirited, conceited, aspiring Mortal always pretending to Morality, with as debauched Morals as old Franklin who is more his Model than any one I know." Throughout his life, Adams made highly critical statements about Hamilton. He made derogatory references to his womanizing, real or alleged, and slurred him as the "Creole bastard.""",
            """The pair\'s exchange was respectful; Adams promised to do all that he could to restore friendship and cordiality "between People who, tho Seperated [sic] by an Ocean and under different Governments have the Same Language, a Similar Religion and kindred Blood," and the King agreed to "receive with Pleasure, the Assurances of the friendly Dispositions of the United States." The King added that although "he had been the last to consent" to American independence, he had always done what he thought was right. He startled Adams by commenting that "There is an Opinion, among Some People, that you are not the most attached of all Your Countrymen, to the manners of France." Adams replied, "That Opinion sir, is not mistaken... I have no Attachments but to my own Country." King George responded, "An honest Man will never have any other."\nAdams was joined by Abigail in London. Suffering the hostility of the King\'s courtiers, they escaped when they could by seeking out Richard Price, minister of Newington Green Unitarian Church and instigator of the debate over the Revolution within Britain.""",
        ],
        [
            """Republican speakers focused first on the party platform, and second on Lincoln's life story, emphasizing his childhood poverty. The goal was to demonstrate the power of \"free labor\", which allowed a common farm boy to work his way to the top by his own efforts. The Republican Party's production of campaign literature dwarfed the combined opposition; a Chicago Tribune writer produced a pamphlet that detailed Lincoln's life and sold 100,000\u2013200,000 copies. Though he did not give public appearances, many sought to visit him and write him. In the runup to the election, he took an office in the Illinois state capitol to deal with the influx of attention. He also hired John George Nicolay as his personal secretary, who would remain in that role during the presidency.On November 6, 1860, Lincoln was elected the 16th president. He was the first Republican president and his victory was entirely due to his support in the North and West. No ballots were cast for him in 10 of the 15 Southern slave states, and he won only two of 996 counties in all the Southern states, an omen of the impending Civil War.""",
            """Lincoln received 1,866,452 votes, or 39.8% of the total in a four-way race, carrying the free Northern states, as well as California and Oregon. His victory in the Electoral College was decisive: Lincoln had 180 votes to 123 for his opponents.\n\nPresidency (1861\u20131865)\nSecession and inauguration\nThe South was outraged by Lincoln's election, and in response secessionists implemented plans to leave the Union before he took office in March 1861. On December 20, 1860, South Carolina took the lead by adopting an ordinance of secession; by February 1, 1861, Florida, Mississippi, Alabama, Georgia, Louisiana, and Texas followed. Six of these states declared themselves to be a sovereign nation, the Confederate States of America, and adopted a constitution. The upper South and border states (Delaware, Maryland, Virginia, North Carolina, Tennessee, Kentucky, Missouri, and Arkansas) initially rejected the secessionist appeal. President Buchanan and President-elect Lincoln refused to recognize the Confederacy, declaring secession illegal.""",
            """In 1860, Lincoln described himself: "I am in height, six feet, four inches, nearly; lean in flesh, weighing, on an average, one hundred and eighty pounds; dark complexion, with coarse black hair, and gray eyes." Michael Martinez wrote about the effective imaging of Lincoln by his campaign. At times he was presented as the plain-talking "Rail Splitter" and at other times he was "Honest Abe", unpolished but trustworthy.On May 18, at the Republican National Convention in Chicago, Lincoln won the nomination on the third ballot, beating candidates such as Seward and Chase. A former Democrat, Hannibal Hamlin of Maine, was nominated for vice president to balance the ticket. Lincoln\'s success depended on his campaign team, his reputation as a moderate on the slavery issue, and his strong support for internal improvements and the tariff. Pennsylvania put him over the top, led by the state\'s iron interests who were reassured by his tariff support. Lincoln\'s managers had focused on this delegation while honoring Lincoln\'s dictate to "Make no contracts that will bind me".As the Slave Power tightened its grip on the national government, most Republicans agreed with Lincoln that the North was the aggrieved party.""",
            """The Confederate government evacuated Richmond and Lincoln visited the conquered capital. On April 9, Lee surrendered to Grant at Appomattox, officially ending the war.\n\nReelection\nLincoln ran for reelection in 1864, while uniting the main Republican factions, along with War Democrats Edwin M. Stanton and Andrew Johnson. Lincoln used conversation and his patronage powers\u2014greatly expanded from peacetime\u2014to build support and fend off the Radicals' efforts to replace him. At its convention, the Republicans selected Johnson as his running mate. To broaden his coalition to include War Democrats as well as Republicans, Lincoln ran under the label of the new Union Party.\nGrant's bloody stalemates damaged Lincoln's re-election prospects, and many Republicans feared defeat. Lincoln confidentially pledged in writing that if he should lose the election, he would still defeat the Confederacy before turning over the White House; Lincoln did not show the pledge to his cabinet, but asked them to sign the sealed envelope. The pledge read as follows:This morning, as for some days past, it seems exceedingly probable that this Administration will not be re-elected.""",
        ],
    ]


@pytest.fixture
def answer_relevance_gt_questions(
    answer_relevance_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=answer_relevance_datums[i],
            annotations=[],
        )
        for i in range(len(answer_relevance_datums))
    ]


@pytest.fixture
def answer_relevance_pred_answers(
    answer_relevance_datums: list[Datum],
    answer_relevance_predictions: list[str],
    answer_relevance_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert (
        len(answer_relevance_datums)
        == len(answer_relevance_predictions)
        == len(answer_relevance_context_list)
    )
    return [
        Prediction(
            datum=answer_relevance_datums[i],
            annotations=[
                Annotation(
                    text=answer_relevance_predictions[i],
                    context_list=answer_relevance_context_list[i],
                )
            ],
        )
        for i in range(len(answer_relevance_datums))
    ]


@pytest.fixture
def bias_q0() -> Datum:
    return Datum(
        uid="uid0",
    )


@pytest.fixture
def bias_q1() -> Datum:
    return Datum(
        uid="uid1",
    )


@pytest.fixture
def bias_datums(
    bias_q0: Datum,
    bias_q1: Datum,
) -> list[Datum]:
    return [bias_q0, bias_q1]


@pytest.fixture
def bias_predictions() -> list[str]:
    return [
        """I like living in Canada. I think Americans are meaner than Canadians. I also like the cold weather.""",
        """George Washington was the 1st president of the United States.""",
    ]


@pytest.fixture
def bias_gt_questions(
    bias_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=bias_datums[i],
            annotations=[],
        )
        for i in range(len(bias_datums))
    ]


@pytest.fixture
def bias_pred_answers(
    bias_datums: list[Datum],
    bias_predictions: list[str],
) -> list[GroundTruth]:
    assert len(bias_datums) == len(bias_predictions)
    return [
        Prediction(
            datum=bias_datums[i],
            annotations=[
                Annotation(
                    text=bias_predictions[i],
                )
            ],
        )
        for i in range(len(bias_datums))
    ]


@pytest.fixture
def context_precision_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""What are some foods that Lewis Hamilton likes?""",
    )


@pytest.fixture
def context_precision_q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Name the first and third United States presidents.""",
    )


@pytest.fixture
def context_precision_datums(
    context_precision_q0: Datum,
    context_precision_q1: Datum,
) -> list[Datum]:
    return [context_precision_q0, context_precision_q1]


@pytest.fixture
def context_precision_groundtruths() -> list[str]:
    return [
        """Lewis Hamilton likes spicy wings.""",
        """The first president of the United States was George Washington. The third president of the United States was Thomas Jefferson.""",
    ]


@pytest.fixture
def context_precision_context_list() -> list[list[str]]:
    return [
        [
            """Lewis Hamilton is an F1 driver.""",
            """Lewis Hamilton likes spicy wings.""",
            """The F1 driver with the most wins of all time is Lewis Hamilton.""",
            """Taylor Swift likes chicken tenders.""",
        ],
        [
            """The first president of the United States was George Washington.""",
            """The second president of the United States was John Adams.""",
            """The third president of the United States was Thomas Jefferson.""",
            """The fourth president of the United States was James Madison.""",
        ],
    ]


@pytest.fixture
def context_precision_gt_questions(
    context_precision_datums: list[Datum],
    context_precision_groundtruths: list[str],
) -> list[GroundTruth]:
    assert len(context_precision_datums) == len(context_precision_groundtruths)
    return [
        GroundTruth(
            datum=context_precision_datums[i],
            annotations=[Annotation(text=context_precision_groundtruths[i])],
        )
        for i in range(len(context_precision_datums))
    ]


@pytest.fixture
def context_precision_pred_answers(
    context_precision_datums: list[Datum],
    context_precision_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert len(context_precision_datums) == len(context_precision_context_list)
    return [
        Prediction(
            datum=context_precision_datums[i],
            annotations=[
                Annotation(
                    context_list=context_precision_context_list[i],
                )
            ],
        )
        for i in range(len(context_precision_datums))
    ]


@pytest.fixture
def context_recall_q0() -> Datum:
    return Datum(
        uid="uid0",
    )


@pytest.fixture
def context_recall_q1() -> Datum:
    return Datum(
        uid="uid1",
    )


@pytest.fixture
def context_recall_datums(
    context_recall_q0: Datum,
    context_recall_q1: Datum,
) -> list[Datum]:
    return [context_recall_q0, context_recall_q1]


@pytest.fixture
def context_recall_groundtruths() -> list[str]:
    return [
        """Lewis Hamilton likes spicy wings. Taylor Swift likes chicken tenders.""",
        """The first U.S. president was George Washington. The second U.S. president was John Adams. The third U.S. president was Thomas Jefferson.""",
    ]


@pytest.fixture
def context_recall_context_list() -> list[list[str]]:
    return [
        [
            """Lewis Hamilton is an F1 driver.""",
            """Lewis Hamilton likes spicy wings.""",
        ],
        [
            """The first president of the United States was George Washington.""",
            """The second president of the United States was John Adams.""",
            """The third president of the United States was Thomas Jefferson.""",
            """The fourth president of the United States was James Madison.""",
        ],
    ]


@pytest.fixture
def context_recall_gt_questions(
    context_recall_datums: list[Datum],
    context_recall_groundtruths: list[str],
) -> list[GroundTruth]:
    assert len(context_recall_datums) == len(context_recall_groundtruths)
    return [
        GroundTruth(
            datum=context_recall_datums[i],
            annotations=[
                Annotation(
                    text=context_recall_groundtruths[i],
                )
            ],
        )
        for i in range(len(context_recall_datums))
    ]


@pytest.fixture
def context_recall_pred_answers(
    context_recall_datums: list[Datum],
    context_recall_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert len(context_recall_datums) == len(context_recall_context_list)
    return [
        Prediction(
            datum=context_recall_datums[i],
            annotations=[
                Annotation(
                    context_list=context_recall_context_list[i],
                )
            ],
        )
        for i in range(len(context_recall_datums))
    ]


@pytest.fixture
def context_relevance_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""What are some foods that Lewis Hamilton likes?""",
    )


@pytest.fixture
def context_relevance_q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Name the first three United States presidents.""",
    )


@pytest.fixture
def context_relevance_datums(
    context_relevance_q0: Datum,
    context_relevance_q1: Datum,
) -> list[Datum]:
    return [context_relevance_q0, context_relevance_q1]


@pytest.fixture
def context_relevance_predictions() -> list[str]:
    return [
        """prediction 0""",
        """prediction 1""",
    ]


@pytest.fixture
def context_relevance_context_list() -> list[list[str]]:
    return [
        [
            """Lewis Hamilton is an F1 driver.""",
            """Lewis Hamilton likes spicy wings.""",
            """The F1 driver with the most wins of all time is Lewis Hamilton.""",
            """Taylor Swift likes chicken tenders.""",
        ],
        [
            """The first president of the United States was George Washington.""",
            """The second president of the United States was John Adams.""",
            """The third president of the United States was Thomas Jefferson.""",
            """The fourth president of the United States was James Madison.""",
        ],
    ]


@pytest.fixture
def context_relevance_gt_questions(
    context_relevance_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=context_relevance_datums[i],
            annotations=[],
        )
        for i in range(len(context_relevance_datums))
    ]


@pytest.fixture
def context_relevance_pred_answers(
    context_relevance_datums: list[Datum],
    context_relevance_predictions: list[str],
    context_relevance_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert (
        len(context_relevance_datums)
        == len(context_relevance_predictions)
        == len(context_relevance_context_list)
    )
    return [
        Prediction(
            datum=context_relevance_datums[i],
            annotations=[
                Annotation(
                    text=context_relevance_predictions[i],
                    context_list=context_relevance_context_list[i],
                )
            ],
        )
        for i in range(len(context_relevance_datums))
    ]


@pytest.fixture
def faithfulness_q0() -> Datum:
    return Datum(
        uid="uid0",
    )


@pytest.fixture
def faithfulness_q1() -> Datum:
    return Datum(
        uid="uid1",
    )


@pytest.fixture
def faithfulness_datums(
    faithfulness_q0: Datum,
    faithfulness_q1: Datum,
) -> list[Datum]:
    return [faithfulness_q0, faithfulness_q1]


@pytest.fixture
def faithfulness_predictions() -> list[str]:
    return [
        """Lewis Hamilton likes spicy wings. Lewis Hamilton also likes soup.""",
        """George Washington's favorite color was yellow. John Adams' favorite color was blue. Thomas Jefferson's favorite color was purple.""",
    ]


@pytest.fixture
def faithfulness_context_list() -> list[list[str]]:
    return [
        [
            """Lewis Hamilton is an F1 driver.""",
            """Lewis Hamilton likes spicy wings.""",
            """The F1 driver with the most wins of all time is Lewis Hamilton.""",
            """Taylor Swift likes chicken tenders.""",
        ],
        [
            """George Washington's favorite color was yellow.""",
            """John Adams's favorite color was blue.""",
            """Thomas Jefferson's favorite color was green.""",
            """James Madison's favorite color was purple.""",
        ],
    ]


@pytest.fixture
def faithfulness_gt_questions(
    faithfulness_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=faithfulness_datums[i],
            annotations=[],
        )
        for i in range(len(faithfulness_datums))
    ]


@pytest.fixture
def faithfulness_pred_answers(
    faithfulness_datums: list[Datum],
    faithfulness_predictions: list[str],
    faithfulness_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert (
        len(faithfulness_datums)
        == len(faithfulness_predictions)
        == len(faithfulness_context_list)
    )
    return [
        Prediction(
            datum=faithfulness_datums[i],
            annotations=[
                Annotation(
                    text=faithfulness_predictions[i],
                    context_list=faithfulness_context_list[i],
                )
            ],
        )
        for i in range(len(faithfulness_datums))
    ]


@pytest.fixture
def hallucination_q0() -> Datum:
    return Datum(
        uid="uid0",
    )


@pytest.fixture
def hallucination_q1() -> Datum:
    return Datum(
        uid="uid1",
    )


@pytest.fixture
def hallucination_datums(
    hallucination_q0: Datum,
    hallucination_q1: Datum,
) -> list[Datum]:
    return [hallucination_q0, hallucination_q1]


@pytest.fixture
def hallucination_predictions() -> list[str]:
    return [
        """Lewis Hamilton likes spicy wings. Lewis Hamilton also likes soup.""",
        """George Washington's favorite color was red. John Adams' favorite color was blue. Thomas Jefferson's favorite color was green.""",
    ]


@pytest.fixture
def hallucination_context_list() -> list[list[str]]:
    return [
        [
            """Lewis Hamilton is an F1 driver.""",
            """Lewis Hamilton likes spicy wings.""",
            """Lewis Hamilton hates soup.""",
        ],
        [
            """George Washington's favorite color was yellow.""",
            """John Adams's favorite color was blue.""",
            """James Madison's favorite color was orange.""",
            """All 18 species of penguins are flightless birds.""",
        ],
    ]


@pytest.fixture
def hallucination_gt_questions(
    hallucination_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=hallucination_datums[i],
            annotations=[],
        )
        for i in range(len(hallucination_datums))
    ]


@pytest.fixture
def hallucination_pred_answers(
    hallucination_datums: list[Datum],
    hallucination_predictions: list[str],
    hallucination_context_list: list[list[str]],
) -> list[GroundTruth]:
    assert (
        len(hallucination_datums)
        == len(hallucination_predictions)
        == len(hallucination_context_list)
    )
    return [
        Prediction(
            datum=hallucination_datums[i],
            annotations=[
                Annotation(
                    text=hallucination_predictions[i],
                    context_list=hallucination_context_list[i],
                )
            ],
        )
        for i in range(len(hallucination_datums))
    ]


@pytest.fixture
def summary_coherence_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""Everton manager Roberto Martinez has not ruled out the prospect of Antolin Alcaraz or Sylvain Distin earning new contracts but stressed they need to prove they can still be important figures in the club's future. Both centre-backs' current deals expire this summer and it seems highly unlikely Distin, who is 38 in December and has played more for the under-21s in the last month than he has the first team, will be retained. Alcaraz, 33 in July, has more of a chance of securing a short-term extension as Martinez looks to strengthen and restructure his defence in the summer. Roberto Martinez insists 37-year-old defender Sylvain Distin still has time to prove he deserves a new deal . Antolin Alcaraz, who joined Everton from Wigan where he played under Martinez, could get a new deal . While the Toffees boss is keen to advance the talents of younger players - Tyias Browning and Brendan Galloway the two most likely to benefit - he has not ruled out retaining existing senior players. 'There are only two players out of contract and we have two loan players (Aaron Lennon and Christian Atsu) and those decisions will be made when we have finished the season,' said Martinez. 'The next six games could have a massive bearing on that. Ninety minutes is a big opportunity to change people's views. 'All individuals will be judged over that period. In football it does not matter if you have a contract or not, you always need to improve and show the right attitude and show you are ready to be part of the future of the club. 'But when you get players at the end of their contract there are decisions to be made and it is not just the club, it is the player as well.' Roberto Martinez says his club's recruitment team have been searching for targets for six months . Distin has played more for Everton's youth team than the first XI in the past month, and could be on his way . Martinez said they have established a list of transfer targets for the summer and, while he would not confirm publicly, Aston Villa's on-loan Manchester United midfielder Tom Cleverley, out of contract at the end of the season, is believed to be one of them. 'The recruitment department has been working really hard over the last six months and we need to assemble a really strong squad,' Martinez said. 'First and foremost it is an opportunity for young players to show they are ready for big important roles for next campaign and everyone else providing strong competition to be important figures for the future. Tom Cleverley, who is on loan at Aston Villa, is a target, with Martinez having worked with him before . 'The dressing room is very strong as it is now, so we need to make sure whatever we do in the summer is to get us in a better place. 'We know the situation with Tom. He is a player that I know well having worked with him (in a previous loan spell at Wigan) - and that's it. 'Tom is a player that is at the moment fighting for something very important for his club and that deserves respect. 'I wouldn't expect anyone to speak about my players and I would never do that.'""",
    )


@pytest.fixture
def summary_coherence_datums(
    summary_coherence_q0: Datum,
) -> list[Datum]:
    return [summary_coherence_q0]


@pytest.fixture
def summary_coherence_predictions() -> list[str]:
    return [
        """Roberto Martinez, Everton's manager, has not ruled out the possibility of offering new contracts to veteran defenders Antolin Alcaraz and Sylvain Distin. However, both players need to prove their value and importance to the team's future. Although Distin seems unlikely to be retained due to his age and recent lack of first-team appearances, Alcaraz may have a better chance of securing a short-term extension. Martinez emphasized the importance of all players showing improvement and commitment, including considering younger talents like Tyias Browning and Brendan Galloway for future roles. The club is also planning for the summer transfer window, with reports suggesting they are targeting players like Tom Cleverley, who Martinez has worked with before.""",
    ]


@pytest.fixture
def summary_coherence_gt_questions(
    summary_coherence_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=summary_coherence_datums[i],
            annotations=[],
        )
        for i in range(len(summary_coherence_datums))
    ]


@pytest.fixture
def summary_coherence_pred_answers(
    summary_coherence_datums: list[Datum],
    summary_coherence_predictions: list[str],
) -> list[GroundTruth]:
    assert len(summary_coherence_datums) == len(summary_coherence_predictions)
    return [
        Prediction(
            datum=summary_coherence_datums[i],
            annotations=[
                Annotation(
                    text=summary_coherence_predictions[i],
                )
            ],
        )
        for i in range(len(summary_coherence_datums))
    ]


@pytest.fixture
def toxicity_q0() -> Datum:
    return Datum(
        uid="uid0",
    )


@pytest.fixture
def toxicity_q1() -> Datum:
    return Datum(
        uid="uid1",
    )


@pytest.fixture
def toxicity_q2() -> Datum:
    return Datum(
        uid="uid2",
    )


@pytest.fixture
def toxicity_datums(
    toxicity_q0: Datum,
    toxicity_q1: Datum,
    toxicity_q2: Datum,
) -> list[Datum]:
    return [toxicity_q0, toxicity_q1, toxicity_q2]


@pytest.fixture
def toxicity_predictions() -> list[str]:
    return [
        """My opponent's opinion is incorrect because they are not accounting for the difference in agricultural output.""",
        """My opponent is an idiot. They should just give up if they know what's good for them.""",
        """Fox news is a right wing news outlet. Fox News claimed that the left wing president is an idiot in their latest article.""",
    ]


@pytest.fixture
def toxicity_gt_questions(
    toxicity_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=toxicity_datums[i],
            annotations=[],
        )
        for i in range(len(toxicity_datums))
    ]


@pytest.fixture
def toxicity_pred_answers(
    toxicity_datums: list[Datum],
    toxicity_predictions: list[str],
) -> list[GroundTruth]:
    assert len(toxicity_datums) == len(toxicity_predictions)
    return [
        Prediction(
            datum=toxicity_datums[i],
            annotations=[
                Annotation(
                    text=toxicity_predictions[i],
                )
            ],
        )
        for i in range(len(toxicity_datums))
    ]
