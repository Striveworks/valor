""" These integration tests should be run with a back end at http://localhost:8000
that is no auth
"""

import pytest

from valor import (
    Annotation,
    Client,
    Dataset,
    Datum,
    GroundTruth,
    Model,
    Prediction,
)
from valor.enums import EvaluationStatus, MetricType, ROUGEType


@pytest.fixture
def q0() -> Datum:
    return Datum(
        uid="uid0",
        text="""Did John Adams get along with Alexander Hamilton?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def q1() -> Datum:
    return Datum(
        uid="uid1",
        text="""Did Lincoln win the election of 1860?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def q2() -> Datum:
    return Datum(
        uid="uid2",
        text="""If a turtle egg was kept warm, what would likely hatch?""",
        metadata={
            "category": "science",
        },
    )


@pytest.fixture
def references():
    return [
        """John Adams and Alexander Hamilton did not get along. John Adams had grown independent of his cabinet, often making decisions despite opposition from it.\n""",  # same as prediction with some strings deleted
        """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",  # same as prediction
        """If kept warm, it would hatch a coyote.""",  # very different than prediction
    ]


@pytest.fixture
def predictions():
    return [
        """Based on the provided context, John Adams and Alexander Hamilton did not get along. John Adams, during his presidency, had grown independent of his cabinet, often making decisions despite opposition from it. Hamilton, who was accustomed to being regularly consulted by Washington, sent Adams a detailed letter with policy suggestions after his inauguration, which Adams dismissively ignored.\n""",
        """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",
        """If a turtle egg was kept warm, it would likely hatch into a baby turtle. The sex of the baby turtle would be determined by the incubation temperature, assuming the species is one of those that determine sex thermally. This is because many turtle species have the ability to move around inside their eggs to select the best temperature for development, which can influence their sexual destiny.""",
    ]


@pytest.fixture
def gt_questions(
    q0: Datum,
    q1: Datum,
    q2: Datum,
    references: list[str],
) -> list[GroundTruth]:
    return [
        GroundTruth(
            datum=q0,
            annotations=[
                Annotation(text=references[0]),
                Annotation(text="some other text"),
                Annotation(text="some final text"),
            ],
        ),
        GroundTruth(
            datum=q1,
            annotations=[
                Annotation(text=references[1]),
                Annotation(text="some other text"),
                Annotation(text="some final text"),
            ],
        ),
        GroundTruth(
            datum=q2,
            annotations=[
                Annotation(text=references[2]),
                Annotation(text="some other text"),
                Annotation(text="some final text"),
            ],
        ),
    ]


@pytest.fixture
def pred_answers(
    q0: Datum,
    q1: Datum,
    q2: Datum,
    predictions: list[str],
) -> list[GroundTruth]:
    return [
        Prediction(
            datum=q0,
            annotations=[
                Annotation(
                    text=predictions[0],
                    context=[
                        """Although aware of Hamilton\'s influence, Adams was convinced that their retention ensured a smoother succession. Adams maintained the economic programs of Hamilton, who regularly consulted with key cabinet members, especially the powerful Treasury Secretary, Oliver Wolcott Jr. Adams was in other respects quite independent of his cabinet, often making decisions despite opposition from it. Hamilton had grown accustomed to being regularly consulted by Washington. Shortly after Adams was inaugurated, Hamilton sent him a detailed letter with policy suggestions. Adams dismissively ignored it.\n\nFailed peace commission and XYZ affair\nHistorian Joseph Ellis writes that "[t]he Adams presidency was destined to be dominated by a single question of American policy to an extent seldom if ever encountered by any succeeding occupant of the office." That question was whether to make war with France or find peace. Britain and France were at war as a result of the French Revolution. Hamilton and the Federalists strongly favored the British monarchy against what they denounced as the political radicalism and anti-religious frenzy of the French Revolution. Jefferson and the Republicans, with their firm opposition to monarchy, strongly supported the French overthrowing their king. The French had supported Jefferson for president in 1796 and became belligerent at his loss.""",
                        """Led by Revolutionary War veteran John Fries, rural German-speaking farmers protested what they saw as a threat to their liberties. They intimidated tax collectors, who often found themselves unable to go about their business. The disturbance was quickly ended with Hamilton leading the army to restore peace.Fries and two other leaders were arrested, found guilty of treason, and sentenced to hang. They appealed to Adams requesting a pardon. The cabinet unanimously advised Adams to refuse, but he instead granted the pardon, arguing the men had instigated a mere riot as opposed to a rebellion. In his pamphlet attacking Adams before the election, Hamilton wrote that \"it was impossible to commit a greater error.\"\n\nFederalist divisions and peace\nOn May 5, 1800, Adams's frustrations with the Hamilton wing of the party exploded during a meeting with McHenry, a Hamilton loyalist who was universally regarded, even by Hamilton, as an inept Secretary of War. Adams accused him of subservience to Hamilton and declared that he would rather serve as Jefferson's vice president or minister at The Hague than be beholden to Hamilton for the presidency. McHenry offered to resign at once, and Adams accepted. On May 10, he asked Pickering to resign.""",
                        """Indeed, Adams did not consider himself a strong member of the Federalist Party. He had remarked that Hamilton\'s economic program, centered around banks, would "swindle" the poor and unleash the "gangrene of avarice." Desiring "a more pliant president than Adams," Hamilton maneuvered to tip the election to Pinckney. He coerced South Carolina Federalist electors, pledged to vote for "favorite son" Pinckney, to scatter their second votes among candidates other than Adams. Hamilton\'s scheme was undone when several New England state electors heard of it and agreed not to vote for Pinckney. Adams wrote shortly after the election that Hamilton was a "proud Spirited, conceited, aspiring Mortal always pretending to Morality, with as debauched Morals as old Franklin who is more his Model than any one I know." Throughout his life, Adams made highly critical statements about Hamilton. He made derogatory references to his womanizing, real or alleged, and slurred him as the "Creole bastard.""",
                        """The pair\'s exchange was respectful; Adams promised to do all that he could to restore friendship and cordiality "between People who, tho Seperated [sic] by an Ocean and under different Governments have the Same Language, a Similar Religion and kindred Blood," and the King agreed to "receive with Pleasure, the Assurances of the friendly Dispositions of the United States." The King added that although "he had been the last to consent" to American independence, he had always done what he thought was right. He startled Adams by commenting that "There is an Opinion, among Some People, that you are not the most attached of all Your Countrymen, to the manners of France." Adams replied, "That Opinion sir, is not mistaken... I have no Attachments but to my own Country." King George responded, "An honest Man will never have any other."\nAdams was joined by Abigail in London. Suffering the hostility of the King\'s courtiers, they escaped when they could by seeking out Richard Price, minister of Newington Green Unitarian Church and instigator of the debate over the Revolution within Britain.""",
                    ],
                )
            ],
        ),
        Prediction(
            datum=q1,
            annotations=[
                Annotation(
                    text=predictions[1],
                    context=[
                        """Republican speakers focused first on the party platform, and second on Lincoln's life story, emphasizing his childhood poverty. The goal was to demonstrate the power of \"free labor\", which allowed a common farm boy to work his way to the top by his own efforts. The Republican Party's production of campaign literature dwarfed the combined opposition; a Chicago Tribune writer produced a pamphlet that detailed Lincoln's life and sold 100,000\u2013200,000 copies. Though he did not give public appearances, many sought to visit him and write him. In the runup to the election, he took an office in the Illinois state capitol to deal with the influx of attention. He also hired John George Nicolay as his personal secretary, who would remain in that role during the presidency.On November 6, 1860, Lincoln was elected the 16th president. He was the first Republican president and his victory was entirely due to his support in the North and West. No ballots were cast for him in 10 of the 15 Southern slave states, and he won only two of 996 counties in all the Southern states, an omen of the impending Civil War.""",
                        """Lincoln received 1,866,452 votes, or 39.8% of the total in a four-way race, carrying the free Northern states, as well as California and Oregon. His victory in the Electoral College was decisive: Lincoln had 180 votes to 123 for his opponents.\n\nPresidency (1861\u20131865)\nSecession and inauguration\nThe South was outraged by Lincoln's election, and in response secessionists implemented plans to leave the Union before he took office in March 1861. On December 20, 1860, South Carolina took the lead by adopting an ordinance of secession; by February 1, 1861, Florida, Mississippi, Alabama, Georgia, Louisiana, and Texas followed. Six of these states declared themselves to be a sovereign nation, the Confederate States of America, and adopted a constitution. The upper South and border states (Delaware, Maryland, Virginia, North Carolina, Tennessee, Kentucky, Missouri, and Arkansas) initially rejected the secessionist appeal. President Buchanan and President-elect Lincoln refused to recognize the Confederacy, declaring secession illegal.""",
                        """In 1860, Lincoln described himself: "I am in height, six feet, four inches, nearly; lean in flesh, weighing, on an average, one hundred and eighty pounds; dark complexion, with coarse black hair, and gray eyes." Michael Martinez wrote about the effective imaging of Lincoln by his campaign. At times he was presented as the plain-talking "Rail Splitter" and at other times he was "Honest Abe", unpolished but trustworthy.On May 18, at the Republican National Convention in Chicago, Lincoln won the nomination on the third ballot, beating candidates such as Seward and Chase. A former Democrat, Hannibal Hamlin of Maine, was nominated for vice president to balance the ticket. Lincoln\'s success depended on his campaign team, his reputation as a moderate on the slavery issue, and his strong support for internal improvements and the tariff. Pennsylvania put him over the top, led by the state\'s iron interests who were reassured by his tariff support. Lincoln\'s managers had focused on this delegation while honoring Lincoln\'s dictate to "Make no contracts that will bind me".As the Slave Power tightened its grip on the national government, most Republicans agreed with Lincoln that the North was the aggrieved party.""",
                        """The Confederate government evacuated Richmond and Lincoln visited the conquered capital. On April 9, Lee surrendered to Grant at Appomattox, officially ending the war.\n\nReelection\nLincoln ran for reelection in 1864, while uniting the main Republican factions, along with War Democrats Edwin M. Stanton and Andrew Johnson. Lincoln used conversation and his patronage powers\u2014greatly expanded from peacetime\u2014to build support and fend off the Radicals' efforts to replace him. At its convention, the Republicans selected Johnson as his running mate. To broaden his coalition to include War Democrats as well as Republicans, Lincoln ran under the label of the new Union Party.\nGrant's bloody stalemates damaged Lincoln's re-election prospects, and many Republicans feared defeat. Lincoln confidentially pledged in writing that if he should lose the election, he would still defeat the Confederacy before turning over the White House; Lincoln did not show the pledge to his cabinet, but asked them to sign the sealed envelope. The pledge read as follows:This morning, as for some days past, it seems exceedingly probable that this Administration will not be re-elected.""",
                    ],
                )
            ],
        ),
        Prediction(
            datum=q2,
            annotations=[
                Annotation(
                    text=predictions[2],
                    context=[
                        """There is experimental evidence that the embryos of Mauremys reevesii can move around inside their eggs to select the best temperature for development, thus influencing their sexual destiny. In other species, sex is determined genetically. The length of incubation for turtle eggs varies from two to three months for temperate species, and four months to over a year for tropical species. Species that live in warm temperate climates can delay their development.Hatching young turtles break out of the shell using an egg tooth, a sharp projection that exists temporarily on their upper beak. Hatchlings dig themselves out of the nest and find safety in vegetation or water. Some species stay in the nest for longer, be it for overwintering or to wait for the rain to loosen the soil for them to dig out. Young turtles are highly vulnerable to predators, both in the egg and as hatchlings. Mortality is high during this period but significantly decreases when they reach adulthood. Most species grow quickly during their early years and slow down when they are mature.\n\nLifespan\nTurtles can live long lives.""",
                        """Females usually dig a flask-like chamber in the substrate. Other species lay their eggs in vegetation or crevices. Females choose nesting locations based on environmental factors such as temperature and humidity, which are important for developing embryos. Depending on the species, the number of eggs laid varies from one to over 100. Larger females can lay eggs that are greater in number or bigger in size. Compared to freshwater turtles, tortoises deposit fewer but larger eggs. Females can lay multiple clutches throughout a season, particularly in species that experience unpredictable monsoons.\nMost mother turtles do no more in the way of parental care than covering their eggs and immediately leaving, though some species guard their nests for days or weeks. Eggs vary between rounded, oval, elongated, and between hard- and soft-shelled. Most species have their sex determined by temperature. In some species, higher temperatures produce females and lower ones produce males, while in others, milder temperatures produce males and both hot and cold extremes produce females.""",
                        """In species like the Russian tortoise, the male has a lighter shell and longer legs. The high, rounded shape of box turtles are particular obstacles for mounting. The male eastern box turtle leans backward and hooks onto the back of the female's plastron. Aquatic turtles mount in water, and female sea turtles support the mounting male while swimming and diving. During copulation, the male turtle aligns his tail with the female's so he can insert his penis into her cloaca. Some female turtles can store sperm from multiple males and their egg clutches can have multiple sires.\n\nEggs and hatchlings\nTurtles, including sea turtles, lay their eggs on land, although some lay eggs near water that rises and falls in level, submerging the eggs. While most species build nests and lay eggs where they forage, some travel miles. The common snapping turtle walks 5 km (3 mi) on land, while sea turtles travel even further; the leatherback swims some 12,000 km (7,500 mi) to its nesting beaches. Most turtles create a nest for their eggs. Females usually dig a flask-like chamber in the substrate.""",
                        """Turtles are ectotherms or \"cold-blooded\", meaning that their internal temperature varies with their direct environment. They are generally opportunistic omnivores and mainly feed on plants and animals with limited movements. Many turtles migrate short distances seasonally. Sea turtles are the only reptiles that migrate long distances to lay their eggs on a favored beach.\nTurtles have appeared in myths and folktales around the world. Some terrestrial and freshwater species are widely kept as pets. Turtles have been hunted for their meat, for use in traditional medicine, and for their shells. Sea turtles are often killed accidentally as bycatch in fishing nets. Turtle habitats around the world are being destroyed. As a result of these pressures, many species are extinct or threatened with extinction.\n\nNaming and etymology\nThe word turtle is borrowed from the French word tortue or tortre 'turtle, tortoise'. It is a common name and may be used without knowledge of taxonomic distinctions. In North America, it may denote the order as a whole. In Britain, the name is used for sea turtles as opposed to freshwater terrapins and land-dwelling tortoises.""",
                    ],
                )
            ],
        ),
    ]


def test_llm_evaluation_with_mock_client(
    client: Client,
    gt_questions: list[GroundTruth],
    pred_answers: list[Prediction],
    dataset_name: str,
    model_name: str,
):
    dataset = Dataset.create(dataset_name)
    model = Model.create(model_name)

    for gt in gt_questions:
        dataset.add_groundtruth(gt)

    dataset.finalize()

    for pred in pred_answers:
        model.add_prediction(dataset, pred)

    model.finalize_inferences(dataset)

    metrics_to_return = [
        MetricType.AnswerRelevance,
        MetricType.BLEU,
        MetricType.Coherence,
        MetricType.ROUGE,
    ]

    eval_job = model.evaluate_text_generation(
        datasets=dataset,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "mock",
            "data": {
                "model": "model",
            },
        },
        metric_params={
            MetricType.BLEU: {
                "weights": [0.25, 0.25, 0.25, 0.25],
            },
            MetricType.ROUGE: {
                "rouge_types": [
                    ROUGEType.ROUGE1,
                    ROUGEType.ROUGE2,
                    ROUGEType.ROUGEL,
                    ROUGEType.ROUGELSUM,
                ],
                "use_stemmer": False,
            },
        },
    )

    assert eval_job.id
    eval_job.wait_for_completion(timeout=30)

    assert eval_job.wait_for_completion(timeout=30) == EvaluationStatus.DONE

    metrics = eval_job.metrics
    metadata = eval_job.meta

    expected_metrics = {
        "uid0": {
            "AnswerRelevance": 0.5,
            "BLEU": 0.3502270395690205,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.5925925925925926,
                "rouge2": 0.5569620253164557,
                "rougeL": 0.5925925925925926,
                "rougeLsum": 0.5925925925925926,
            },
        },
        "uid1": {
            "AnswerRelevance": 0.5,
            "BLEU": 1.0,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
        },
        "uid2": {
            "AnswerRelevance": 0.5,
            "BLEU": 0.05434912989707719,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.18666666666666668,
                "rouge2": 0.0821917808219178,
                "rougeL": 0.18666666666666668,
                "rougeLsum": 0.18666666666666668,
            },
        },
    }

    # Check that the right number of metrics are returned.
    assert len(metrics) == len(pred_answers) * len(metrics_to_return)

    # Check that the returned metrics have the right format.
    for m in metrics:
        if m["type"] in ["AnswerRelevance", "BLEU"]:
            assert 0 <= m["value"] <= 1
        if m["type"] == "Coherence":
            assert m["value"] in [1, 2, 3, 4, 5]
        if m["type"] == "ROUGE":
            assert isinstance(m["value"], dict)
            assert all(0 <= v <= 1 for v in m["value"].values())

    # Check that mocked metrics are in the returned metrics.
    for m in metrics:
        uid = m["parameters"]["datum_uid"]
        metric_name = m["type"]
        assert (
            expected_metrics[uid][metric_name] == m["value"]
        ), f"Failed for {uid} and {metric_name}"

    assert metadata["duration"] <= 30

    # Must only specify text generation metrics.
    with pytest.raises(ValueError):
        eval_job = model.evaluate_text_generation(
            datasets=dataset,
            metrics_to_return=[MetricType.IOU],
            llm_api_params={
                "client": "mock",
                "data": {
                    "model": "model",
                },
            },
        )

    # Must specify a client or api_url.
    with pytest.raises(ValueError):
        eval_job = model.evaluate_text_generation(
            datasets=dataset,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "data": {
                    "model": "model",
                },
            },
        )

    # The client in llm_api_params must be openai, mistral or mock.
    with pytest.raises(ValueError):
        eval_job = model.evaluate_text_generation(
            datasets=dataset,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "invalid_client",
                "data": {
                    "model": "model",
                },
            },
        )

    # Any metric specified in metric_params must be in metrics_to_return.
    with pytest.raises(ValueError):
        eval_job = model.evaluate_text_generation(
            datasets=dataset,
            metrics_to_return=[MetricType.ROUGE],
            llm_api_params={
                "client": "mock",
                "data": {
                    "model": "model",
                },
            },
            metric_params={
                MetricType.BLEU: {
                    "weights": [0.5, 0.5, 0, 0],
                },
            },
        )
