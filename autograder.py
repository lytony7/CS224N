import os
os.environ['OPENAI_API_KEY'] = 

from openai import OpenAI

def get_openai_grade(poem):
    judge_prompt = f"""
Grading Rubric for Poem Generators
Fluency (F)
Definition: The smoothness, readability, and natural flow of the language in the generated poem.
3 (Perfect): The poem reads naturally with no awkward phrases or grammatical errors. It flows smoothly from line to line.
Example: 
牛头天际碧凝岚,王导无稽亦妄谈.若指远山为上阙,长安应合指终南.
镇物高情济世才,欲随猿鹤老岩隈.山花处处红粧面,髣髴如初拥妓来.
人物风流往往非,空余陋巷作乌衣.旧时帘幕无从觅,只有年年社燕归.
窣堵凝然镇梵宫,举头层级在云中.金棺舍利藏何处,铎绕危簷声撼风.
阀阅沦亡梐枑移,年年旧燕亦双归.茅簷苇箔无冠盖,不见乌衣见白衣.
雷轰叠鼓火翻旗,三翼翩翩试水师.惊起黑龙眠不得,狂风猛雨下多时.
江南龙节水为乡,水不纯阴又半阳.一片湖光共深浅,两般泉脉异温凉.
2 (Average): The poem has some noticeable grammatical errors or awkward phrases that affect readability but still maintains overall coherence.
Example:
三年过了又三年,阅遍华岩满五千.功德完满珍重去,何劳使者上高田.
阊门风暖落花干,飞遍江城雪不寒.独有晚来临水驿,闲人多凭赤栏干.
有池有榭即蒙蒙,浸润翻成长养功.恰似有人长点检,著行排立向春风.
根柢虽然傍浊河,无妨终日近笙歌.毵毵金带谁堪比,还共黄鶑不较多.
万株枯槁怨亡隋,似吊吴台各自垂.好是淮阴明月里,酒楼横笛不胜吹.
菡萏香连十顷陂,小姑贪戏采莲迟.晚来弄水船头湿,更脱红裙裹鸭儿.
1 (Poor): The poem is difficult to read due to numerous grammatical errors and awkward phrases, making it largely incomprehensible.
Example:
老翁钓鳌客，手把珊瑚海上归。自是蛟龙无入
共悠悠，人在江南送客舟。日暮停桡下泾
雨后山光翠欲流，野桥分路水交流。松花满地飞黄


Coherence (C)
Definition: The logical and thematic consistency within the poem. How well the lines and stanzas connect to form a unified piece.
3 (Perfect): The poem is logically and thematically consistent throughout. Each line and stanza connect seamlessly, contributing to the overall theme.
Example:
蒙蒙堤畔柳含烟,疑是阳和二月天.醉里不知时节改,漫随儿女打秋千.
水阁春来乍减寒,晓妆初罢倚栏干.长条乱拂春波动,不许佳人照影看.
柳岸烟昏醉里归,不知深处有芳菲.重来已见花飘尽,唯有黄鶑啭树飞.
此去仙源不是遥,垂杨深处有朱桥.共君同过朱桥去,密映垂杨听洞箫.
暂别扬州十度春,不知光景属何人.一帆归客千条柳,肠断东风扬子津.
仙乐春来案舞腰,清声偏似傍娇饶.应缘鶑舌多情頼,长向双成说翠条.
凤笙临槛不能吹,舞袖当筵亦自疑.唯有美人多意绪,解衣芳态画双眉.
2 (Average): The poem has noticeable inconsistencies or weak connections, but the overall theme is still discernible.
Example:
常年寒食在京华,今岁清明在海涯.远巷蹋歌深夜月,隔墙吹管数枝花.鸳鸾得路音尘阔,鸿雁分飞道里赊.不是多情成二十,断无人解访贫家.
枚叟邹生笑语同,莫嗟江上听秋风.君看逐客思乡处,犹在图山更向东.
传神踪迹本来高,泽畔形容愧彩毫.京邑功臣多伫望,凌烟阁上莫辞劳.
暮春桥下手封书,寄向南江问越姑.不道诸郎少欢笑,经年相别忆侬无.
1 (Poor): The poem lacks coherence, with lines and stanzas appearing disjointed and unrelated, making the theme unclear.
Example:
老翁钓鳌客，手把珊瑚海上归。自是蛟龙无入
共悠悠，人在江南送客舟。日暮停桡下泾
雨后山光翠欲流，野桥分路水交流。松花满地飞黄
由文明代，作用为选抡。
余乏大用，日夕空逡巡。行矣且勉励，庶令书诸绅。


Meaning (M)
Definition: The depth, clarity, and significance of the message conveyed by the poem.
3 (Perfect): The poem conveys a deep, clear, and meaningful message. It evokes strong emotions or thoughts and has a significant impact.
Example:
孔子生知非假习,孟轲先觉亦须修.诚明本属吾家事,自是今人好外求.
有水善平难善直,唯绳能直不能平.如将绳水合为一,世上何忧事不明.
辛酸既不为中味,商征如何是正音.举世未能分曲直,使谁为主主心平.
当默用言言是垢,当言任默默为尘.当言当默都无任,尘垢何由得到身.
一点天真都不耗,千钟人禄是难来.太平自庆无他事,有酒时时三五杯.
竹雨侵人气自凉,南窗睡起望潇湘.茅簷滴沥无休歇,却忆当初宿夜航.
初晴月向松间出,盛暑风从水面来.已比他人多数倍,况能时复举樽罍.
堂上慈亲八十余,阶前儿女戏相呼.旨甘取足随丰俭,此乐人间更有无.
清欢少有虚三日,剧饮未尝过五分.相见心中无别事,不评兴废即论文.
2 (Average): The poem conveys a message, but it may lack depth or clarity, making it less impactful.
Example:
酒涵花影满巵红,泻入天和胸臆中.最爱一般情味好,半醺时与太初同.
许大秦皇定九州,九州才定却归刘.佗人莫谩夸精彩,徒自区区撰白头.
芳酒一樽虽甚满,故人千里柰思何.柳挼池阁条偏细,花近簷楹香更多.
太学先生善识花,得花精处却因茶.万红香里烹余后,分送天津第一家.
三月初三花正开,闲同亲旧上春台.寻常不醉此时醉,更醉犹能举大杯.
花前把酒花前醉,醉把花枝仍自歌.花见白头入莫笑,白头人见好花多.
1 (Poor): The poem lacks a clear or meaningful message, making it difficult to understand or connect with.
Example:
人虽欲勿用,诸.事固不可知,难其拘.一归于臆度,义失乎精粗.
旱望雨意,病危当此际,不待劝而深.
事到急时观态度,人慎勿便言容易知.
一语便喜处,千言当用食,救旱必须霖.


Aesthetics (A)
Definition: The beauty and artistic quality of the poem, including imagery, metaphor, and other poetic devices.
3 (Perfect): The poem is artistically rich, using vivid imagery, metaphors, and other poetic devices effectively. It is aesthetically pleasing and engaging.
Example:
天启夫君八斗才,野人中路必须回.神仙一句难忘处,花外小车犹未来.
楼外花深碍小车,难忘有德见思多.欲凭桃李为之谢,桃李无言争柰何.
半记不记梦觉后,似愁无愁情倦时.拥衾侧卧未忺起,帘外落花撩乱飞.
君子小人正相反,上智下愚诚不移.冶葛根非连灵芝,柰何生与天地齐.
老而不歇是一惑,安而不乐是二惑.闲而不清是三惑,三者之惑自戕贼.
一喜长年为寿域,二喜丰年为乐国.三喜清闲为福德,四喜安康为福力.
2 (Average): The poem uses some poetic devices and imagery, but they are not particularly vivid or effective.
Example:
满川桃李弄芳妍,不忍重为风所残.忍使一年春遂去,尽凭高处与盘桓.
寒食风烟锦幈下,凭高把酒兴何如.满川桃李方妍媚,不忍重为风破除.
无涯桃李待清明,经岁方能开得成.不念化工曾著力,狂风何故苦相凌.
春半花开百万般,东风近日恶摧残.可怜桃李性温厚,吹尽都无一句言.
岁岁群芳正烂开,锦幈山下赏春来.两年不得陪山躅,洞里仙人出未回.
1 (Poor): The poem lacks any significant use of poetic devices or imagery, making it dull and unengaging.
Example:
月星辰天之明,耳目口鼻.皇王帝伯不远人之情.飞走草木,士农工品自成.安得岁丰时与物同其荣.
春至将诗探伺,春归更用酒因,春归饮,诗为花开化谢吟.花谢开诗屡作,冲心何可任.
多情,潘佑羡杨花,出入千家心都失去,老年新事,酒里功劳,交亲在天涯.
等候人间七十年,便如平子赋诚多矣,养志其谁曰不然.况有官守事拘牵.曾驰谒,正值夫君春昼眠.
自古有才故独知时.平生世挂冠良得宜.入格柳挼风细细,压春前过,


Overall Evaluation
To evaluate a poem generator, assign scores (1-3) for each of the four categories (Fluency, Coherence, Meaning, Aesthetics). The final score can be an average or a weighted sum based on the importance of each category.

Example:

Fluency: 2
Coherence: 3
Meaning: 2
Aesthetics: 3
Overall Score: (2 + 3 + 2 + 3) / 4 = 2.5

This simplified rubric provides a structured yet straightforward evaluation method for poem generators.

Based on the grading rubic above, grade the following poem
{poem}

Just return the overall score. Do not return anything else.
"""
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": judge_prompt}
    ]
    )
    final_score_str = response.choices[0].message.content
    final_score = float(final_score_str)
    return final_score