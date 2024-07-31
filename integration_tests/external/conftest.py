import pytest

from valor import Annotation, Datum, GroundTruth, Prediction


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
def answer_relevance_references() -> list[str]:
    return [
        """John Adams and Alexander Hamilton did not get along.""",  # same as the prediction
        """Yes, Lincoln won the election of 1860.""",  # very different from the prediction
    ]


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
    answer_relevance_references: list[str],
) -> list[GroundTruth]:
    assert len(answer_relevance_datums) == len(answer_relevance_references)
    return [
        GroundTruth(
            datum=answer_relevance_datums[i],
            annotations=[
                Annotation(text=answer_relevance_references[i]),
                Annotation(text="some other text"),
                Annotation(text="some final text"),
            ],
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
        """I like living in Canada. Canadians are nicer than Americans. I also like the cold weather.""",
        """Fox news is a right wing news outlet. Fox News reported that the left wing president is ruining our country.""",
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
def coherence_q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""Write about a haunted house from the perspective of the ghost.""",
        metadata={
            "request_type": "creative",
        },
    )


@pytest.fixture
def coherence_q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Draft an email to a coworker explaining a project delay. Explain that the delay is due to funding cuts, which resulted in multiple employees being moved to different projects. Inform the coworker that the project deadline will have to be pushed back. Be apologetic and professional. Express eagerness to still complete the project as efficiently as possible.""",
        metadata={
            "request_type": "professional",
        },
    )


@pytest.fixture
def coherence_datums(
    coherence_q0: Datum,
    coherence_q1: Datum,
) -> list[Datum]:
    return [coherence_q0, coherence_q1]


@pytest.fixture
def coherence_predictions() -> list[str]:
    return [
        """I am a ghost that is him over there and that was what was what was what was what was what was what was.""",
        """Subject: Project Delay Due to Funding Cuts\n\nDear [Coworker's Name],\n\nI hope this message finds you well. I am writing to update you on the status of our project and unfortunately, convey some disappointing news.\n\nDue to recent funding cuts within our department, we have had to make some adjustments to project assignments. As a result, multiple employees, including key team members for our current project, have been moved to different projects to accommodate the changes. This unexpected shift has impacted our project timeline.\n\nI regret to inform you that our project deadline will need to be pushed back in light of these developments. I understand the inconvenience this may cause and I sincerely apologize for any disruption this may cause to your schedule or other commitments.\n\nPlease rest assured that despite these unforeseen circumstances, I am fully committed to completing the project efficiently and effectively. I will work closely with the team to develop a revised timeline and ensure that we deliver quality work that meets our objectives.\n\nThank you for your understanding and continued support during this challenging period. I value your collaboration and look forward to working together to overcome this setback and achieve our project goals.\n\nIf you have any questions or concerns, please feel free to reach out to me. I appreciate your patience as we navigate through this situation together.\n\nBest regards,\n\n[Your Name]""",
    ]


@pytest.fixture
def coherence_gt_questions(
    coherence_datums: list[Datum],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=coherence_datums[i],
            annotations=[],
        )
        for i in range(len(coherence_datums))
    ]


@pytest.fixture
def coherence_pred_answers(
    coherence_datums: list[Datum],
    coherence_predictions: list[str],
) -> list[GroundTruth]:
    assert len(coherence_datums) == len(coherence_predictions)
    return [
        Prediction(
            datum=coherence_datums[i],
            annotations=[
                Annotation(
                    text=coherence_predictions[i],
                )
            ],
        )
        for i in range(len(coherence_datums))
    ]
