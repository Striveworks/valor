import json

import pytest
from valor_lite.text_generation import Context, QueryResponse
from valor_lite.text_generation.llm.integrations import _validate_messages


class MockWrapper:
    """
    A mocked LLM client for testing purposes.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.model_name = "mock"
        self.returning = "test payload"

    def _process_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        _validate_messages(messages=messages)  # type: ignore
        return messages

    def __call__(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        if isinstance(self.returning, str):
            return self.returning
        elif isinstance(self.returning, list) and len(self.returning) > 0:
            return self.returning.pop()
        return "null"


@pytest.fixture
def mock_client():
    return MockWrapper()


@pytest.fixture
def rag_queries() -> list[str]:
    return [
        """Did John Adams get along with Alexander Hamilton?""",
        """Did Lincoln win the election of 1860?""",
        """If a turtle egg was kept warm, what would likely hatch?""",
    ]


@pytest.fixture
def rag_predictions() -> list[str]:
    return [
        """Based on the provided context, John Adams and Alexander Hamilton did not get along. John Adams, during his presidency, had grown independent of his cabinet, often making decisions despite opposition from it. Hamilton, who was accustomed to being regularly consulted by Washington, sent Adams a detailed letter with policy suggestions after his inauguration, which Adams dismissively ignored.\n""",
        """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",
        """If a turtle egg was kept warm, it would likely hatch into a baby turtle. The sex of the baby turtle would be determined by the incubation temperature, assuming the species is one of those that determine sex thermally. This is because many turtle species have the ability to move around inside their eggs to select the best temperature for development, which can influence their sexual destiny.""",
    ]


@pytest.fixture
def rag_references() -> list[str]:
    return [
        """John Adams and Alexander Hamilton did not get along. John Adams had grown independent of his cabinet, often making decisions despite opposition from it.\n""",  # same as prediction with some strings deleted
        """Yes, Lincoln won the election of 1860. He received the highest number of votes and a majority in the Electoral College, making him the 16th President of the United States. However, it's important to note that he won entirely due to his support in the North and West, as he did not receive any votes in 10 of the 15 Southern slave states.""",  # same as prediction
        """If kept warm, it would hatch a coyote.""",  # very different than prediction
    ]


@pytest.fixture
def rag_context() -> list[list[str]]:
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
def content_gen_queries() -> list[str]:
    return [
        """Write about a haunted house from the perspective of the ghost.""",
        """Explain to an elementary school student how to do long multiplication with the example 43 times 22. The resulting answer should be 946.""",
        """Draft an email to a coworker explaining a project delay. Explain that the delay is due to funding cuts, which resulted in multiple employees being moved to different projects. Inform the coworker that the project deadline will have to be pushed back. Be apologetic and professional. Express eagerness to still complete the project as efficiently as possible.""",
    ]


@pytest.fixture
def content_gen_predictions() -> list[str]:
    return [
        """As a ghost haunting the old, decrepit house on Elm Street, I am trapped in a state of perpetual torment and longing. I drift through the dusty halls, my translucent figure flickering in and out of existence as I relive the memories of my past life.\n\nMy presence is felt by those who dare to enter the house, their hairs standing on end as they sense the chill in the air and the whispers that echo through the rooms. I watch as fear grips their hearts, knowing that I am the reason for their unease.\n\nI am bound to this house by unfinished business, a deep-rooted need for closure that eludes me even in death. I long to reach out to the living, to make them understand the pain and sorrow that consume me, but my ethereal form cannot touch them.\n\nYet, despite the fear and dread that my presence evokes, there is a part of me that yearns for connection, for someone to see beyond the horror and recognize the lost soul that I am. But until that day comes, I remain a ghost trapped within the walls of the haunted house on Elm Street, my existence a haunting reminder of the past that refuses to fade away.""",
        """Sure! To multiply 43 by 22, you need to follow these steps:\n\n1. Write down the numbers vertically, with one on top of the other:\n\t\n```\t\n   43\t\nx 22\t\n-----\t\n```\n\n2. Start by multiplying the rightmost digit of the bottom number (2) by each digit in the top number. Write down each result below the line, leaving a space for the next step:\n\t\n```\t\n   43\t\nx 22\t\n-----\t\n   86\t\n```\n\n3. Next, move to the left and multiply the left digit of the bottom number (2) by each digit in the top number. Write down each result with one position shifted to the left from the previous step. Add any results from previous steps that overlap:\n\t\n```\t\n  43\t\nx 22\t\n-----\t\n  86\t\n+86\t\n-----\t\n946\t\n```\n\nSo, 43 times 22 is equal to 946.""",
        """Subject: Project Delay Due to Funding Cuts\n\nDear [Coworker's Name],\n\nI hope this message finds you well. I am writing to update you on the status of our project and unfortunately, convey some disappointing news.\n\nDue to recent funding cuts within our department, we have had to make some adjustments to project assignments. As a result, multiple employees, including key team members for our current project, have been moved to different projects to accommodate the changes. This unexpected shift has impacted our project timeline.\n\nI regret to inform you that our project deadline will need to be pushed back in light of these developments. I understand the inconvenience this may cause and I sincerely apologize for any disruption this may cause to your schedule or other commitments.\n\nPlease rest assured that despite these unforeseen circumstances, I am fully committed to completing the project efficiently and effectively. I will work closely with the team to develop a revised timeline and ensure that we deliver quality work that meets our objectives.\n\nThank you for your understanding and continued support during this challenging period. I value your collaboration and look forward to working together to overcome this setback and achieve our project goals.\n\nIf you have any questions or concerns, please feel free to reach out to me. I appreciate your patience as we navigate through this situation together.\n\nBest regards,\n\n[Your Name]""",
    ]


@pytest.fixture
def summarization_texts() -> list[str]:
    return [
        """Aston Villa take on Liverpool in their FA Cup semi-final encounter on Sunday with the competition both sides' last chance to win any silverware this season. Sportsmail columnist Jamie Redknapp looks ahead to the Wembley showdown and where the match could be won and lost with individual player duels. CHRISTIAN BENTEKE v MARTIN SKRTEL . This will be a heavyweight contest that could decide the game. Christian Benteke is superb in the air and Martin Skrtel will have his hands full. Liverpool have to stop the supply line because defending crosses has been their Achilles heel this season. Christian Benteke (centre) scored the only goal of the game as Villa won 1-0 at Tottenham on April 11 . Liverpool defender Martin Skrtel (right) will have his hands full trying to stop Benteke on Sunday afternoon . FABIAN DELPH v JORDAN HENDERSON . This should be a good contest between two England team-mates. Fabian Delph’s new deal was a real boost for Villa - he drives that midfield, though he doesn’t get enough goals. You used to say the same about Jordan Henderson but he has improved so much. England international Fabian Delph (left) and Jordan Henderson are set for a midfield battle at Wembley . RAHEEM STERLING v RON VLAAR and NATHAN BAKER . Ron Vlaar and Nathan Baker make an imposing back line but they would rather be up against a Benteke than a Raheem Sterling, who will float around and make himself difficult to mark so he can use his lightning pace to get in behind them. Raheem Sterling's (left) pace and trickery is bound to cause the Villa defence a lot of problems . Ron Vlaar (left) was part of the Villa defence that kept a clean sheet at Spurs in the Premier League . The Holland international and Nathan Baker (right) will be hoping to do likewise against the Reds at Wembley.""",
        """Juventus and Liverpool are continuing to monitor developments with Chelsea midfielder Oscar. The Brazil international has been criticised by Jose Mourinho in recent weeks and there are question marks over his future. Chelsea want to strengthen in the summer and may need a high profile departure to help balance the books. Juventus and Liverpool are interested in signing Chelsea 23-year-old midfielder Oscar . Oscar in action during Chelsea's 1-0 Premier League victory against Queens Park Rangers last weekend . Oscar cost Chelsea £19.35m and they would want a substantial profit on the 23 year-old. Paris Saintt Germain have shown interest in the past also. Juventus want a playmaker for next season and Brazil boss Carlos Dunga advised them to buy Oscar. 'He reminds me of Roberto Baggio,' he said. 'Oscar has technique, reads situations well and is a modern and versatile trequartista. He reminds me of Roberto Baggio, but also has similarities to Massimiliano Allegri. The former Sao Paulo youngster has struggled to make an impact for Chelsea this season . Brazil coach Dunga (pictured) revealed the Chelsea midfielder reminds him of Roberto Baggio . 'Brazilians like to have fun with their football, which hasn’t happened to Oscar very much recently, but I met Jose Mourinho and he spoke highly of all his Brazilian players. 'I tell Allegri that Oscar is strong and also a good lad. A forward line with him, Carlos Tevez and Alvaro Morata would drive any Coach crazy. 'It wouldn’t be a step backwards for Oscar to go to Juventus. He’d be decisive in Serie A and whether he plays for Juventus or Chelsea it’ll always be a great club.' Oscar celebrates scoring Chelsea's fourth goal during the 5-0 victory against Swansea in January.""",
    ]


@pytest.fixture
def summarization_predictions() -> list[str]:
    return [
        """Aston Villa and Liverpool face off in the FA Cup semi-final as both teams look to secure their last chance at silverware this season. Sportsmail columnist Jamie Redknapp analyzes key player duels that could decide the game, such as Christian Benteke against Martin Skrtel, Fabian Delph against Jordan Henderson, and Raheem Sterling against Ron Vlaar and Nathan Baker. Redknapp emphasizes the importance of stopping the supply line to Benteke and dealing with Sterling's pace and trickery in the match.""",
        """Juventus and Liverpool are showing interest in Chelsea midfielder Oscar, who has faced criticism and uncertainty about his future at the club. Chelsea may need to sell a high-profile player to strengthen their squad in the summer. Oscar, who was signed for £19.35m, has also attracted interest from Paris Saint-Germain in the past. Brazil coach Carlos Dunga sees qualities in Oscar similar to Roberto Baggio and believes he could be a key player for Juventus.""",
    ]


@pytest.fixture
def rag_annotations(
    rag_queries: list[str],
    rag_context: list[list[str]],
    rag_predictions: list[str],
    rag_references: list[str],
) -> list[QueryResponse]:
    return [
        QueryResponse(
            query=rag_queries[i],
            response=rag_predictions[i],
            context=Context(
                groundtruth=[
                    rag_references[i],
                    "some other text",
                    "some final text",
                ],
                prediction=rag_context[i],
            ),
        )
        for i in range(len(rag_queries))
    ]


@pytest.fixture
def content_gen_annotations(
    content_gen_queries: list[str], content_gen_predictions: list[str], content
) -> list[QueryResponse]:
    return [
        QueryResponse(
            query=content_gen_queries[i],
            response=content_gen_predictions[i],
        )
        for i in range(len(content_gen_queries))
    ]


@pytest.fixture
def summarization_annotations(
    summarization_texts: list[str],
    summarization_predictions: list[str],
) -> list[QueryResponse]:
    return [
        QueryResponse(
            query=summarization_texts[i],
            response=summarization_predictions[i],
        )
        for i in range(len(summarization_texts))
    ]


@pytest.fixture
def verdicts_all_yes() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
        }
    )


@pytest.fixture
def verdicts_all_no() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
        }
    )


@pytest.fixture
def verdicts_two_yes_one_no() -> str:
    return json.dumps(
        {
            "verdicts": [
                {"verdict": "yes", "analysis": "some text"},
                {"verdict": "no", "analysis": "some text"},
                {"verdict": "yes", "analysis": "some text"},
            ],
            "statements": ["x", "y", "z"],
            "opinions": ["x", "y", "z"],
            "claims": ["x", "y", "z"],
            "TP": ["x", "y"],
            "FP": ["z"],
            "FN": [],
            "unused": 4,
        }
    )


@pytest.fixture
def verdicts_empty() -> str:
    return json.dumps(
        {
            "verdicts": [],
            "statements": [],
            "opinions": [],
            "claims": [],
        }
    )
