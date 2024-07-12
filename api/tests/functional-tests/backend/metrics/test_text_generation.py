from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from valor_api import crud, schemas
from valor_api.backend import models
from valor_api.backend.core import create_or_get_evaluations
from valor_api.backend.metrics.text_generation import (
    _calculate_rouge_scores,
    _calculate_sentence_bleu,
    _compute_text_generation_metrics,
    compute_text_generation_metrics,
)
from valor_api.enums import EvaluationStatus, MetricType, ROUGEType, TaskType

RAG_PREDICTIONS = [
    """Based on the provided context, John Adams and Alexander Hamilton did not get along. John Adams, during his presidency, had grown independent of his cabinet, often making decisions despite opposition from it. Hamilton, who was accustomed to being regularly consulted by Washington, sent Adams a detailed letter with policy suggestions after his inauguration, which Adams dismissively ignored.\n""",
    """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",
    """If a turtle egg was kept warm, it would likely hatch into a baby turtle. The sex of the baby turtle would be determined by the incubation temperature, assuming the species is one of those that determine sex thermally. This is because many turtle species have the ability to move around inside their eggs to select the best temperature for development, which can influence their sexual destiny.""",
]


CONTENT_GEN_PREDICTIONS = [
    """As a ghost haunting the old, decrepit house on Elm Street, I am trapped in a state of perpetual torment and longing. I drift through the dusty halls, my translucent figure flickering in and out of existence as I relive the memories of my past life.\n\nMy presence is felt by those who dare to enter the house, their hairs standing on end as they sense the chill in the air and the whispers that echo through the rooms. I watch as fear grips their hearts, knowing that I am the reason for their unease.\n\nI am bound to this house by unfinished business, a deep-rooted need for closure that eludes me even in death. I long to reach out to the living, to make them understand the pain and sorrow that consume me, but my ethereal form cannot touch them.\n\nYet, despite the fear and dread that my presence evokes, there is a part of me that yearns for connection, for someone to see beyond the horror and recognize the lost soul that I am. But until that day comes, I remain a ghost trapped within the walls of the haunted house on Elm Street, my existence a haunting reminder of the past that refuses to fade away.""",
    """Sure! To multiply 43 by 22, you need to follow these steps:\n\n1. Write down the numbers vertically, with one on top of the other:\n\t\n```\t\n   43\t\nx 22\t\n-----\t\n```\n\n2. Start by multiplying the rightmost digit of the bottom number (2) by each digit in the top number. Write down each result below the line, leaving a space for the next step:\n\t\n```\t\n   43\t\nx 22\t\n-----\t\n   86\t\n```\n\n3. Next, move to the left and multiply the left digit of the bottom number (2) by each digit in the top number. Write down each result with one position shifted to the left from the previous step. Add any results from previous steps that overlap:\n\t\n```\t\n  43\t\nx 22\t\n-----\t\n  86\t\n+86\t\n-----\t\n946\t\n```\n\nSo, 43 times 22 is equal to 946.""",
    """Subject: Project Delay Due to Funding Cuts\n\nDear [Coworker's Name],\n\nI hope this message finds you well. I am writing to update you on the status of our project and unfortunately, convey some disappointing news.\n\nDue to recent funding cuts within our department, we have had to make some adjustments to project assignments. As a result, multiple employees, including key team members for our current project, have been moved to different projects to accommodate the changes. This unexpected shift has impacted our project timeline.\n\nI regret to inform you that our project deadline will need to be pushed back in light of these developments. I understand the inconvenience this may cause and I sincerely apologize for any disruption this may cause to your schedule or other commitments.\n\nPlease rest assured that despite these unforeseen circumstances, I am fully committed to completing the project efficiently and effectively. I will work closely with the team to develop a revised timeline and ensure that we deliver quality work that meets our objectives.\n\nThank you for your understanding and continued support during this challenging period. I value your collaboration and look forward to working together to overcome this setback and achieve our project goals.\n\nIf you have any questions or concerns, please feel free to reach out to me. I appreciate your patience as we navigate through this situation together.\n\nBest regards,\n\n[Your Name]""",
]


@pytest.fixture
def rag_dataset_name() -> str:
    return "rag_test_dataset"


@pytest.fixture
def rag_model_name() -> str:
    return "rag_test_model"


@pytest.fixture
def rag_q0() -> schemas.Datum:
    return schemas.Datum(
        uid="uid0",
        text="""Did John Adams get along with Alexander Hamilton?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def rag_q1() -> schemas.Datum:
    return schemas.Datum(
        uid="uid1",
        text="""Did Lincoln win the election of 1860?""",
        metadata={
            "category": "history",
        },
    )


@pytest.fixture
def rag_q2() -> schemas.Datum:
    return schemas.Datum(
        uid="uid2",
        text="""If a turtle egg was kept warm, what would likely hatch?""",
        metadata={
            "category": "science",
        },
    )


@pytest.fixture
def rag_references():
    return [
        """John Adams and Alexander Hamilton did not get along. John Adams had grown independent of his cabinet, often making decisions despite opposition from it.\n""",  # same as prediction with some strings deleted
        """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",  # same as prediction
        """If kept warm, it would hatch a coyote.""",  # very different than prediction
    ]


@pytest.fixture
def rag_context_per_prediction():
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
        [
            """There is experimental evidence that the embryos of Mauremys reevesii can move around inside their eggs to select the best temperature for development, thus influencing their sexual destiny. In other species, sex is determined genetically. The length of incubation for turtle eggs varies from two to three months for temperate species, and four months to over a year for tropical species. Species that live in warm temperate climates can delay their development.Hatching young turtles break out of the shell using an egg tooth, a sharp projection that exists temporarily on their upper beak. Hatchlings dig themselves out of the nest and find safety in vegetation or water. Some species stay in the nest for longer, be it for overwintering or to wait for the rain to loosen the soil for them to dig out. Young turtles are highly vulnerable to predators, both in the egg and as hatchlings. Mortality is high during this period but significantly decreases when they reach adulthood. Most species grow quickly during their early years and slow down when they are mature.\n\nLifespan\nTurtles can live long lives.""",
            """Females usually dig a flask-like chamber in the substrate. Other species lay their eggs in vegetation or crevices. Females choose nesting locations based on environmental factors such as temperature and humidity, which are important for developing embryos. Depending on the species, the number of eggs laid varies from one to over 100. Larger females can lay eggs that are greater in number or bigger in size. Compared to freshwater turtles, tortoises deposit fewer but larger eggs. Females can lay multiple clutches throughout a season, particularly in species that experience unpredictable monsoons.\nMost mother turtles do no more in the way of parental care than covering their eggs and immediately leaving, though some species guard their nests for days or weeks. Eggs vary between rounded, oval, elongated, and between hard- and soft-shelled. Most species have their sex determined by temperature. In some species, higher temperatures produce females and lower ones produce males, while in others, milder temperatures produce males and both hot and cold extremes produce females.""",
            """In species like the Russian tortoise, the male has a lighter shell and longer legs. The high, rounded shape of box turtles are particular obstacles for mounting. The male eastern box turtle leans backward and hooks onto the back of the female's plastron. Aquatic turtles mount in water, and female sea turtles support the mounting male while swimming and diving. During copulation, the male turtle aligns his tail with the female's so he can insert his penis into her cloaca. Some female turtles can store sperm from multiple males and their egg clutches can have multiple sires.\n\nEggs and hatchlings\nTurtles, including sea turtles, lay their eggs on land, although some lay eggs near water that rises and falls in level, submerging the eggs. While most species build nests and lay eggs where they forage, some travel miles. The common snapping turtle walks 5 km (3 mi) on land, while sea turtles travel even further; the leatherback swims some 12,000 km (7,500 mi) to its nesting beaches. Most turtles create a nest for their eggs. Females usually dig a flask-like chamber in the substrate.""",
            """Turtles are ectotherms or \"cold-blooded\", meaning that their internal temperature varies with their direct environment. They are generally opportunistic omnivores and mainly feed on plants and animals with limited movements. Many turtles migrate short distances seasonally. Sea turtles are the only reptiles that migrate long distances to lay their eggs on a favored beach.\nTurtles have appeared in myths and folktales around the world. Some terrestrial and freshwater species are widely kept as pets. Turtles have been hunted for their meat, for use in traditional medicine, and for their shells. Sea turtles are often killed accidentally as bycatch in fishing nets. Turtle habitats around the world are being destroyed. As a result of these pressures, many species are extinct or threatened with extinction.\n\nNaming and etymology\nThe word turtle is borrowed from the French word tortue or tortre 'turtle, tortoise'. It is a common name and may be used without knowledge of taxonomic distinctions. In North America, it may denote the order as a whole. In Britain, the name is used for sea turtles as opposed to freshwater terrapins and land-dwelling tortoises.""",
        ],
    ]


@pytest.fixture
def rag_data(
    db: Session,
    rag_dataset_name: str,
    rag_model_name: str,
    rag_q0: schemas.Datum,
    rag_q1: schemas.Datum,
    rag_q2: schemas.Datum,
    rag_references: list[str],
    rag_context_per_prediction: list[list[str]],
):
    datums = [rag_q0, rag_q1, rag_q2]

    gts = []
    for i in range(len(datums)):
        gts.append(
            schemas.GroundTruth(
                dataset_name=rag_dataset_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(text=rag_references[i]),
                    schemas.Annotation(text="some other text"),
                    schemas.Annotation(text="some final text"),
                ],
            )
        )

    preds = []
    for i in range(len(datums)):
        preds.append(
            schemas.Prediction(
                dataset_name=rag_dataset_name,
                model_name=rag_model_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(
                        text=RAG_PREDICTIONS[i],
                        context=rag_context_per_prediction[i],
                    )
                ],
            )
        )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=rag_dataset_name,
            metadata={"type": "text"},
        ),
    )

    crud.create_groundtruths(db=db, groundtruths=gts)
    crud.finalize(db=db, dataset_name=rag_dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=rag_model_name,
            metadata={
                "type": "text",
                "hf_model_name": """mistralai/Mixtral-8x7B-Instruct-v0.1""",
                "raw_text_field": "context",
                "input": """{context}\n{question}""",
                "prompt": """Answer the following question with the provided context. The format will be first the context, second the question, third the answer.\n{input}\nAnswer:""",
                "max_new_tokens": 100,
            },
        ),
    )
    crud.create_predictions(db=db, predictions=preds)
    crud.finalize(
        db=db, dataset_name=rag_dataset_name, model_name=rag_model_name
    )

    assert len(db.query(models.Datum).all()) == 3
    assert (
        len(db.query(models.Annotation).all()) == 12
    )  # 3 groundtruths with 3 annotations, 3 predictions with 1 annotation
    assert len(db.query(models.Label).all()) == 0
    assert len(db.query(models.GroundTruth).all()) == 9
    assert len(db.query(models.Prediction).all()) == 3


@pytest.fixture
def content_gen_dataset_name() -> str:
    return "content_gen_test_dataset"


@pytest.fixture
def content_gen_model_name() -> str:
    return "content_gen_test_model"


@pytest.fixture
def content_gen_q0() -> schemas.Datum:
    return schemas.Datum(
        uid="uid0",
        text="""Write about a haunted house from the perspective of the ghost.""",
        metadata={
            "request_type": "creative",
        },
    )


@pytest.fixture
def content_gen_q1() -> schemas.Datum:
    return schemas.Datum(
        uid="uid1",
        text="""Explain to an elementary school student how to do long multiplication with the example 43 times 22. The resulting answer should be 946.""",
        metadata={
            "request_type": "educational",
        },
    )


@pytest.fixture
def content_gen_q2() -> schemas.Datum:
    return schemas.Datum(
        uid="uid2",
        text="""Draft an email to a coworker explaining a project delay. Explain that the delay is due to funding cuts, which resulted in multiple employees being moved to different projects. Inform the coworker that the project deadline will have to be pushed back. Be apologetic and professional. Express eagerness to still complete the project as efficiently as possible.""",
        metadata={
            "request_type": "professional",
        },
    )


@pytest.fixture
def content_gen_data(
    db: Session,
    content_gen_dataset_name: str,
    content_gen_model_name: str,
    content_gen_q0: schemas.Datum,
    content_gen_q1: schemas.Datum,
    content_gen_q2: schemas.Datum,
):
    datums = [content_gen_q0, content_gen_q1, content_gen_q2]

    gts = []
    for i in range(len(datums)):
        gts.append(
            schemas.GroundTruth(
                dataset_name=content_gen_dataset_name,
                datum=datums[i],
                annotations=[],
            )
        )

    preds = []
    for i in range(len(datums)):
        preds.append(
            schemas.Prediction(
                dataset_name=content_gen_dataset_name,
                model_name=content_gen_model_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(
                        text=CONTENT_GEN_PREDICTIONS[i],
                    )
                ],
            )
        )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=content_gen_dataset_name,
            metadata={"type": "text"},
        ),
    )

    crud.create_groundtruths(db=db, groundtruths=gts)
    crud.finalize(db=db, dataset_name=content_gen_dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=content_gen_model_name,
            metadata={
                "type": "text",
            },
        ),
    )
    crud.create_predictions(db=db, predictions=preds)
    crud.finalize(
        db=db,
        dataset_name=content_gen_dataset_name,
        model_name=content_gen_model_name,
    )

    assert len(db.query(models.Datum).all()) == 3
    assert len(db.query(models.Annotation).all()) == 6
    assert len(db.query(models.Label).all()) == 0
    assert len(db.query(models.GroundTruth).all()) == 3
    assert len(db.query(models.Prediction).all()) == 3


@pytest.fixture
def two_text_generation_datasets(
    db: Session,
    rag_dataset_name: str,
    rag_model_name: str,
    rag_q0: schemas.Datum,
    rag_q1: schemas.Datum,
    rag_q2: schemas.Datum,
    rag_references: list[str],
    rag_context_per_prediction: list[list[str]],
    content_gen_dataset_name: str,
    content_gen_model_name: str,
    content_gen_q0: schemas.Datum,
    content_gen_q1: schemas.Datum,
    content_gen_q2: schemas.Datum,
):
    # RAG dataset and model
    datums = [rag_q0, rag_q1, rag_q2]

    gts = []
    for i in range(len(datums)):
        gts.append(
            schemas.GroundTruth(
                dataset_name=rag_dataset_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(text=rag_references[i]),
                    schemas.Annotation(text="some other text"),
                    schemas.Annotation(text="some final text"),
                ],
            )
        )

    preds = []
    for i in range(len(datums)):
        preds.append(
            schemas.Prediction(
                dataset_name=rag_dataset_name,
                model_name=rag_model_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(
                        text=RAG_PREDICTIONS[i],
                        context=rag_context_per_prediction[i],
                    )
                ],
            )
        )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=rag_dataset_name,
            metadata={"type": "text"},
        ),
    )

    crud.create_groundtruths(db=db, groundtruths=gts)
    crud.finalize(db=db, dataset_name=rag_dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=rag_model_name,
            metadata={
                "type": "text",
                "hf_model_name": """mistralai/Mixtral-8x7B-Instruct-v0.1""",
                "raw_text_field": "context",
                "input": """{context}\n{question}""",
                "prompt": """Answer the following question with the provided context. The format will be first the context, second the question, third the answer.\n{input}\nAnswer:""",
                "max_new_tokens": 100,
            },
        ),
    )
    crud.create_predictions(db=db, predictions=preds)
    crud.finalize(
        db=db, dataset_name=rag_dataset_name, model_name=rag_model_name
    )

    # Content generation dataset and model
    datums = [content_gen_q0, content_gen_q1, content_gen_q2]

    gts = []
    for i in range(len(datums)):
        gts.append(
            schemas.GroundTruth(
                dataset_name=content_gen_dataset_name,
                datum=datums[i],
                annotations=[],
            )
        )

    preds = []
    for i in range(len(datums)):
        preds.append(
            schemas.Prediction(
                dataset_name=content_gen_dataset_name,
                model_name=content_gen_model_name,
                datum=datums[i],
                annotations=[
                    schemas.Annotation(
                        text=CONTENT_GEN_PREDICTIONS[i],
                    )
                ],
            )
        )

    crud.create_dataset(
        db=db,
        dataset=schemas.Dataset(
            name=content_gen_dataset_name,
            metadata={"type": "text"},
        ),
    )

    crud.create_groundtruths(db=db, groundtruths=gts)
    crud.finalize(db=db, dataset_name=content_gen_dataset_name)

    crud.create_model(
        db=db,
        model=schemas.Model(
            name=content_gen_model_name,
            metadata={
                "type": "text",
            },
        ),
    )
    crud.create_predictions(db=db, predictions=preds)
    crud.finalize(
        db=db,
        dataset_name=content_gen_dataset_name,
        model_name=content_gen_model_name,
    )

    assert len(db.query(models.Datum).all()) == 6
    assert len(db.query(models.Annotation).all()) == 18
    assert len(db.query(models.Label).all()) == 0
    assert len(db.query(models.GroundTruth).all()) == 12
    assert len(db.query(models.Prediction).all()) == 6


def mocked_connection(self):
    pass


def mocked_answer_relevance(
    self,
    query: str,
    text: str,
):
    ret_dict = {
        RAG_PREDICTIONS[0]: 0.6666666666666666,
        RAG_PREDICTIONS[1]: 0.2,
        RAG_PREDICTIONS[2]: 0.2,
    }
    return ret_dict[text]


def mocked_coherence(
    self,
    text: str,
):
    ret_dict = {
        RAG_PREDICTIONS[0]: 4,
        RAG_PREDICTIONS[1]: 5,
        RAG_PREDICTIONS[2]: 4,
        CONTENT_GEN_PREDICTIONS[0]: 5,
        CONTENT_GEN_PREDICTIONS[1]: 5,
        CONTENT_GEN_PREDICTIONS[2]: 5,
    }
    return ret_dict[text]


def mocked_compute_rouge_none(*args, **kwargs):
    """
    Dummy docstring
    """
    return None


@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.answer_relevance",
    mocked_answer_relevance,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.coherence",
    mocked_coherence,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedMistralAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedMistralAIClient.answer_relevance",
    mocked_answer_relevance,
)
def test__compute_text_generation_rag(
    db: Session,
    rag_dataset_name: str,
    rag_model_name: str,
    rag_data,
):
    """
    Tests the _compute_text_generation function.
    """

    datum_filter = schemas.Filter(
        datasets=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.DATASET_NAME,
                    ),
                    rhs=schemas.Value.infer(rag_dataset_name),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        ),
        models=schemas.LogicalFunction(
            args=[
                schemas.Condition(
                    lhs=schemas.Symbol(
                        name=schemas.SupportedSymbol.MODEL_NAME,
                    ),
                    rhs=schemas.Value.infer(rag_model_name),
                    op=schemas.FilterOperator.EQ,
                ),
            ],
            op=schemas.LogicalOperator.AND,
        ),
    )
    groundtruth_filter = datum_filter.model_copy()
    prediction_filter = datum_filter.model_copy()

    metrics_to_return = [
        MetricType.AnswerRelevance,
        MetricType.Coherence,
        MetricType.ROUGE,
        MetricType.BLEU,
    ]

    metrics = _compute_text_generation_metrics(
        db,
        datum_filter=datum_filter,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "openai",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
    )

    expected_values = {
        "uid0": {
            schemas.AnswerRelevanceMetric: 0.6666666666666666,
            schemas.CoherenceMetric: 4,
            schemas.ROUGEMetric: {
                "rouge1": 0.5925925925925926,
                "rouge2": 0.5569620253164557,
                "rougeL": 0.5925925925925926,
                "rougeLsum": 0.5925925925925926,
            },
            schemas.BLEUMetric: 0.3502270395690205,
        },
        "uid1": {
            schemas.AnswerRelevanceMetric: 0.2,
            schemas.CoherenceMetric: 5,
            schemas.ROUGEMetric: {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            schemas.BLEUMetric: 1.0,
        },
        "uid2": {
            schemas.AnswerRelevanceMetric: 0.2,
            schemas.CoherenceMetric: 4,
            schemas.ROUGEMetric: {
                "rouge1": 0.18666666666666668,
                "rouge2": 0.0821917808219178,
                "rougeL": 0.18666666666666668,
                "rougeLsum": 0.18666666666666668,
            },
            schemas.BLEUMetric: 0.05434912989707719,
        },
    }

    assert metrics
    for metric in metrics:
        assert isinstance(metric.parameters, dict)
        assert isinstance(metric.parameters["datum_uid"], str)
        assert (
            expected_values[metric.parameters["datum_uid"]].get(type(metric))
            == metric.value
        )

    # Test that mistral is accepted as a valid client.
    _ = _compute_text_generation_metrics(
        db,
        datum_filter=datum_filter,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=[MetricType.AnswerRelevance],
        llm_api_params={
            "client": "mistral",
            "data": {
                "model": "mistral-small-latest",
            },
        },
        metric_params={
            "BLEU": {
                "weights": [0.5, 0.25, 0.25, 0],
            },
            "ROUGE": {
                "rouge_types": [
                    ROUGEType.ROUGE1,
                    ROUGEType.ROUGE2,
                    ROUGEType.ROUGEL,
                ],
                "use_stemmer": True,
            },
        },
    )

    # Test that manually specifying the api key works.
    _ = _compute_text_generation_metrics(
        db,
        datum_filter=datum_filter,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "openai",
            "api_key": "test_key",
            "data": {
                "seed": 2024,
                "model": "gpt-4o",
            },
        },
    )

    # Test the mock client.
    _ = _compute_text_generation_metrics(
        db,
        datum_filter=datum_filter,
        groundtruth_filter=groundtruth_filter,
        prediction_filter=prediction_filter,
        metrics_to_return=metrics_to_return,
        llm_api_params={
            "client": "mock",
            "data": {
                "model": "some model",
            },
        },
    )

    # Need to specify the client or api_url (api_url has not been implemented)
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        )

    # Cannot specify both a client and api_url.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "api_url": "openai.com",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        )

    # Support is not implemented for api_url.
    with pytest.raises(NotImplementedError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "api_url": "openai.com",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        )

    # Test that an invalid client raises an error.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "invalid_client",
                "data": {
                    "model": "model",
                },
            },
        )

    # data should be a dictionary.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": "gpt-4o",
            },
        )

    # BLEU metric parameters should be a dictionary.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
            metric_params={
                "BLEU": [0.25, 0.25, 0.25, 0.25],
            },
        )

    # ROUGE metric parameters should be a dictionary.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
            metric_params={
                "ROUGE": ["use_stemmer"],
            },
        )

    # If an llm-guided metric is requested, then llm_api_params must be specified.
    with pytest.raises(ValueError):
        _compute_text_generation_metrics(
            db,
            datum_filter=datum_filter,
            groundtruth_filter=groundtruth_filter,
            prediction_filter=prediction_filter,
            metrics_to_return=metrics_to_return,
        )


@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.answer_relevance",
    mocked_answer_relevance,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.coherence",
    mocked_coherence,
)
def test_text_generation_rag(
    db: Session,
    rag_dataset_name: str,
    rag_model_name: str,
    rag_data,
):
    metrics_to_return = [
        MetricType.AnswerRelevance,
        MetricType.Coherence,
        MetricType.ROUGE,
        MetricType.BLEU,
    ]

    # default request
    job_request = schemas.EvaluationRequest(
        dataset_names=[rag_dataset_name],
        model_names=[rag_model_name],
        parameters=schemas.EvaluationParameters(
            task_type=TaskType.TEXT_GENERATION,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
            bleu_weights=[0.25, 0.25, 0.25, 0.25],
            rouge_types=[
                ROUGEType.ROUGE1,
                ROUGEType.ROUGE2,
                ROUGEType.ROUGEL,
                ROUGEType.ROUGELSUM,
            ],
            rouge_use_stemmer=False,
        ),
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_text_generation_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics

    expected_values = {
        "uid0": {
            "AnswerRelevance": 0.6666666666666666,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.5925925925925926,
                "rouge2": 0.5569620253164557,
                "rougeL": 0.5925925925925926,
                "rougeLsum": 0.5925925925925926,
            },
            "BLEU": 0.3502270395690205,
        },
        "uid1": {
            "AnswerRelevance": 0.2,
            "Coherence": 5,
            "ROUGE": {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            "BLEU": 1.0,
        },
        "uid2": {
            "AnswerRelevance": 0.2,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.18666666666666668,
                "rouge2": 0.0821917808219178,
                "rougeL": 0.18666666666666668,
                "rougeLsum": 0.18666666666666668,
            },
            "BLEU": 0.05434912989707719,
        },
    }

    assert metrics
    for metric in metrics:
        assert isinstance(metric.parameters, dict)
        assert (
            expected_values[metric.parameters["datum_uid"]][metric.type]
            == metric.value
        )

    # Check that specifying rouge_use_stemmer still works even if rouge_types is not supplied.
    job_request = schemas.EvaluationRequest(
        dataset_names=[rag_dataset_name],
        model_names=[rag_model_name],
        parameters=schemas.EvaluationParameters(
            task_type=TaskType.TEXT_GENERATION,
            metrics_to_return=[MetricType.ROUGE],
            rouge_use_stemmer=True,
        ),
    )
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == EvaluationStatus.PENDING
    _ = compute_text_generation_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )


@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.answer_relevance",
    mocked_answer_relevance,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.coherence",
    mocked_coherence,
)
def test_text_generation_content_gen(
    db: Session,
    content_gen_dataset_name: str,
    content_gen_model_name: str,
    content_gen_data,
):
    metrics_to_return = [
        MetricType.Coherence,
    ]

    # default request
    job_request = schemas.EvaluationRequest(
        dataset_names=[content_gen_dataset_name],
        model_names=[content_gen_model_name],
        parameters=schemas.EvaluationParameters(
            task_type=TaskType.TEXT_GENERATION,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        ),
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_text_generation_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics

    expected_values = {
        "uid0": {
            "Coherence": 5,
        },
        "uid1": {
            "Coherence": 5,
        },
        "uid2": {
            "Coherence": 5,
        },
    }

    assert metrics
    for metric in metrics:
        assert isinstance(metric.parameters, dict)
        assert (
            expected_values[metric.parameters["datum_uid"]][metric.type]
            == metric.value
        )


@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.connect",
    mocked_connection,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.answer_relevance",
    mocked_answer_relevance,
)
@patch(
    "valor_api.backend.core.llm_clients.WrappedOpenAIClient.coherence",
    mocked_coherence,
)
def test_text_generation_two_datasets(
    db: Session,
    rag_dataset_name: str,
    rag_model_name: str,
    content_gen_dataset_name: str,
    content_gen_model_name: str,
    two_text_generation_datasets,
):
    # test with a RAG dataset
    metrics_to_return = [
        MetricType.AnswerRelevance,
        MetricType.Coherence,
        MetricType.ROUGE,
        MetricType.BLEU,
    ]

    # default request
    job_request = schemas.EvaluationRequest(
        dataset_names=[rag_dataset_name],
        model_names=[rag_model_name],
        parameters=schemas.EvaluationParameters(
            task_type=TaskType.TEXT_GENERATION,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
            bleu_weights=[0.25, 0.25, 0.25, 0.25],
            rouge_types=[
                ROUGEType.ROUGE1,
                ROUGEType.ROUGE2,
                ROUGEType.ROUGEL,
                ROUGEType.ROUGELSUM,
            ],
            rouge_use_stemmer=False,
        ),
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_text_generation_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics

    expected_values = {
        "uid0": {
            "AnswerRelevance": 0.6666666666666666,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.5925925925925926,
                "rouge2": 0.5569620253164557,
                "rougeL": 0.5925925925925926,
                "rougeLsum": 0.5925925925925926,
            },
            "BLEU": 0.3502270395690205,
        },
        "uid1": {
            "AnswerRelevance": 0.2,
            "Coherence": 5,
            "ROUGE": {
                "rouge1": 1.0,
                "rouge2": 1.0,
                "rougeL": 1.0,
                "rougeLsum": 1.0,
            },
            "BLEU": 1.0,
        },
        "uid2": {
            "AnswerRelevance": 0.2,
            "Coherence": 4,
            "ROUGE": {
                "rouge1": 0.18666666666666668,
                "rouge2": 0.0821917808219178,
                "rougeL": 0.18666666666666668,
                "rougeLsum": 0.18666666666666668,
            },
            "BLEU": 0.05434912989707719,
        },
    }

    assert metrics
    for metric in metrics:
        assert isinstance(metric.parameters, dict)
        assert (
            expected_values[metric.parameters["datum_uid"]][metric.type]
            == metric.value
        )

    # test with a content generation dataset
    metrics_to_return = [
        MetricType.Coherence,
    ]

    # default request
    job_request = schemas.EvaluationRequest(
        dataset_names=[content_gen_dataset_name],
        model_names=[content_gen_model_name],
        parameters=schemas.EvaluationParameters(
            task_type=TaskType.TEXT_GENERATION,
            metrics_to_return=metrics_to_return,
            llm_api_params={
                "client": "openai",
                "data": {
                    "seed": 2024,
                    "model": "gpt-4o",
                },
            },
        ),
    )

    # creates evaluation job
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status == EvaluationStatus.PENDING

    # computation, normally run as background task
    _ = compute_text_generation_metrics(
        db=db,
        evaluation_id=evaluations[0].id,
    )

    # get evaluations
    evaluations = create_or_get_evaluations(db=db, job_request=job_request)
    assert len(evaluations) == 1
    assert evaluations[0].status in {
        EvaluationStatus.RUNNING,
        EvaluationStatus.DONE,
    }

    metrics = evaluations[0].metrics

    expected_values = {
        "uid0": {
            "Coherence": 5,
        },
        "uid1": {
            "Coherence": 5,
        },
        "uid2": {
            "Coherence": 5,
        },
    }

    assert metrics
    for metric in metrics:
        assert isinstance(metric.parameters, dict)
        assert (
            expected_values[metric.parameters["datum_uid"]][metric.type]
            == metric.value
        )


def test__calculate_rouge_scores():
    examples = [
        {
            "prediction": "Mary loves Joe",
            "references": [
                "Mary loves Joe",
            ],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match
        {
            "prediction": "MARY LOVES JOE",
            "references": ["Mary loves Joe"],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "rouge1": 1.0,
            "rouge2": 1.0,
            "rougeL": 1.0,
            "rougeLsum": 1.0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "rouge1": 0.67,
            "rouge2": 0.5,
            "rougeL": 0.67,
            "rougeLsum": 0.67,
        },  # off by one
        {
            "prediction": "flipping the roaring white dolphin",
            "references": ["flip the roaring white dolphin"],
            "rouge1": 0.8,
            "rouge2": 0.75,
            "rougeL": 0.8,
            "rougeLsum": 0.8,
            "use_stemmer": False,
        },  # incorrect match without stemming
        {
            "prediction": "flipping the roaring white dolphin",
            "references": ["flip the roaring white dolphin"],
            "rouge1": 1,
            "rouge2": 1,
            "rougeL": 1,
            "rougeLsum": 1,
            "use_stemmer": True,
        },  # correct match with stemming
        {
            "prediction": "flipping the roaring white dolphin",
            "references": [
                "some random sentence",
                "some other sentence",
                "some final reference",
                "flip the roaring white dolphin",
            ],
            "rouge1": 1,
            "rouge2": 1,
            "rougeL": 1,
            "rougeLsum": 1,
            "use_stemmer": True,
        },  # test multiple references
    ]

    multiple_prediction_examples = [
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": [
                ["Mary loves June", "some other sentence"],
                ["some other sentence", "the big fox hunts rabbits"],
            ],
            "expected_value": [
                {
                    "prediction": "Mary loves Joe",
                    "value": {
                        "rouge1": 0.6666666666666666,
                        "rouge2": 0.5,
                        "rougeL": 0.6666666666666666,
                        "rougeLsum": 0.6666666666666666,
                    },
                },
                {
                    "prediction": "Mary loves Jack",
                    "value": {
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "rougeLsum": 0.0,
                    },
                },
            ],
        },  # off by one
        {
            "prediction": [
                "flipping the roaring white dolphin",
                "Mary loves Joe",
            ],
            "references": [
                [
                    "some random sentence",
                    "some other sentence",
                    "some final reference",
                    "flip the roaring white dolphin",
                ],
                ["beep bop", "Mary loves June"],
            ],
            "expected_value": [
                {
                    "prediction": "flipping the roaring white dolphin",
                    "value": {
                        "rouge1": 1.0,
                        "rouge2": 1.0,
                        "rougeL": 1.0,
                        "rougeLsum": 1.0,
                    },
                },
                {
                    "prediction": "Mary loves Joe",
                    "value": {
                        "rouge1": 0.6666666666666666,
                        "rouge2": 0.5,
                        "rougeL": 0.6666666666666666,
                        "rougeLsum": 0.6666666666666666,
                    },
                },
            ],
            "use_stemmer": True,
        },  # test multiple references and multiple predictions
    ]

    expected_errors = [
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": [["Mary loves June"]],
            "error": ValueError,
            "weights": (1,),
        },  # mismatched predictions and references
        {
            "prediction": ["Mary loves Joe", "Mary loves Jack"],
            "references": ["Mary loves June"],
            "error": ValueError,
        },  # incorrect use of multiple predictions
        {
            "prediction": "Mary loves Joe",
            "references": "Mary loves Joe",
            "weights": (1,),
            "error": ValueError,
        },  # references isn't a list
        {
            "prediction": None,
            "references": "Mary loves Joe",
            "weights": (1,),
            "error": ValueError,
        },  # prediction shouldn't be None
        {
            "prediction": "Mary loves Joe",
            "references": None,
            "weights": (1,),
            "error": ValueError,
        },  # references shouldn't be None
        {
            "prediction": 123,
            "references": None,
            "weights": (1,),
            "error": ValueError,
        },  # prediction must be str or list
    ]

    # test single prediction examples
    for example in examples:
        output = _calculate_rouge_scores(
            predictions=example["prediction"],
            references=example["references"],
            use_stemmer=example.get("use_stemmer", False),
        )[0]
        assert all(
            round(output["value"][key], 2) == example[key]
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        ), f"Error for example {example} with output {output}."

    # test multiple prediction examples
    for example in multiple_prediction_examples:
        metrics = _calculate_rouge_scores(
            predictions=example["prediction"],
            references=example["references"],
            use_stemmer=example.get("use_stemmer", False),
        )
        assert metrics == example["expected_value"]

    for example in expected_errors:
        with pytest.raises(example["error"]):
            _calculate_rouge_scores(
                predictions=example["prediction"],
                references=example["references"],
            )


@patch(
    "evaluate.EvaluationModule.compute",
    mocked_compute_rouge_none,
)
def test__calculate_rouge_scores_with_none():
    prediction = "Mary loves Joe"
    references = ["Mary loves Joe"]

    with pytest.raises(ValueError):
        _calculate_rouge_scores(
            predictions=prediction,
            references=references,
        )


def test__calculate_bleu_scores():
    examples = [
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Joe"],
            "weights": (1,),
            "expected_value": 1.0,
        },  # perfect match
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Joe"],
            "weights": [
                1,
            ],
            "expected_value": 1.0,
        },  # perfect match, weights are a list
        {
            "prediction": "MARY LOVES JOE",
            "references": ["Mary loves Joe"],
            "weights": (1,),
            "expected_value": 0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "weights": (1,),
            "expected_value": 0,
        },  # perfect match, case sensitive
        {
            "prediction": "Mary loves Joe",
            "references": ["MARY LOVES JOE"],
            "weights": (0, 1),
            "expected_value": 0,
        },  # perfect match, case sensitive, BLEU-2
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Joe"],
            "weights": (0, 1),
            "expected_value": 1.0,
        },  # BLEU-2
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Joe"],
            "weights": [0.25] * 4,
            "expected_value": 0,
        },  # BLEU-4
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (1,),
            "expected_value": 0.67,
        },  # off by one
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 1),
            "expected_value": 0.5,
        },  # off by one BLEU-2
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 0, 1),
            "expected_value": 0,
        },  # off by one BLEU-3
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Jane"],
            "weights": (0, 0, 0, 1),
            "expected_value": 0,
        },  # off by one BLEU-4
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": (1,),
            "expected_value": 0,
        },  # different cases
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": [0, 1],
            "expected_value": 0,
        },  # different cases BLEU-2
        {
            "prediction": "mary loves joe",
            "references": ["MARY LOVES JOE"],
            "weights": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "expected_value": 0,
        },  # different cases BLEU-10
        {
            "prediction": "flip the roaring white dolphin",
            "references": [
                "some random sentence",
                "some other sentence",
                "some final reference",
                "flip the roaring white dolphin",
            ],
            "weights": [0, 1],
            "expected_value": 1,
        },  # test multiple references
    ]

    expected_errors = [
        {
            "prediction": "Mary loves Joe",
            "references": "Mary loves Joe",
            "weights": (1,),
            "error": ValueError,
        },  # references isn't a list
        {
            "prediction": None,
            "references": "Mary loves Joe",
            "weights": (1,),
            "error": ValueError,
        },  # prediction shouldn't be None
        {
            "prediction": "Mary loves Joe",
            "references": None,
            "weights": (1,),
            "error": ValueError,
        },  # references shouldn't be None
        {
            "prediction": "Mary loves Joe",
            "references": ["Mary loves Joe"],
            "weights": None,
            "error": ValueError,
        },  # weights shouldn't be None
        {
            "prediction": 0.3,
            "references": ["Mary loves Joe"],
            "weights": (1,),
            "error": ValueError,
        },  # prediction should be a string or list of strings
    ]

    for example in examples:
        output = _calculate_sentence_bleu(
            predictions=example["prediction"],
            references=example["references"],
            weights=example["weights"],
        )
        assert (
            round(output[0]["value"], 2) == example["expected_value"]
        ), f"Error for example {example} with output {output}."

    for example in expected_errors:
        with pytest.raises(example["error"]):
            _calculate_sentence_bleu(
                predictions=example["prediction"],
                references=example["references"],
                weights=example["weights"],
            )
