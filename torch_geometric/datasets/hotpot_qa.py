"""datasets['train'][0]
{'id': '5a7a06935542990198eaf050', 'question': "Which magazine was started first Arthur's Magazine or First for Women?", 'answer': "Arthur's Magazine", 'type': 'comparison', 'level': 'medium', 'supporting_facts': {'title': ["Arthur's Magazine", 'First for Women'], 'sent_id': [0, 0]}, 'context': {'title': ['Radio City (Indian radio station)', 'History of Albanian football', 'Echosmith', "Women's colleges in the Southern United States", 'First Arthur County Courthouse and Jail', "Arthur's Magazine", '2014–15 Ukrainian Hockey Championship', 'First for Women', 'Freeway Complex Fire', 'William Rast'], 'sentences': [["Radio City is India's first private FM radio station and was started on 3 July 2001.", ' It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).', ' It plays Hindi, English and regional songs.', ' It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.', ' Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.', ' The Radio station currently plays a mix of Hindi and Regional music.', ' Abraham Thomas is the CEO of the company.'], ['Football in Albania existed before the Albanian Football Federation (FSHF) was created.', " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .", ' Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.', ' In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA.'], ['Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.', ' Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.', ' Echosmith started first as "Ready Set Go!"', ' until they signed to Warner Bros.', ' Records in May 2012.', ' They are best known for their hit song "Cool Kids", which reached number 13 on the "Billboard" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.', ' The song was Warner Bros.', " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.", ' The band\'s debut album, "Talking Dreams", was released on October 8, 2013.'], ["Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.", " Many started first as girls' seminaries or academies.", ' Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.', ' Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level.'], ['The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum.'], ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.", ' Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.', ' In May 1846 it was merged into "Godey\'s Lady\'s Book".'], ['The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.', ' Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.', ' Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.', ' The regular season included just 12 rounds, where all the teams went to the semifinals.', ' In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk.'], ["First for Women is a woman's magazine published by Bauer Media Group in the USA.", ' The magazine was started in 1989.', ' It is based in Englewood Cliffs, New Jersey.', ' In 2011 the circulation of the magazine was 1,310,696 copies.'], ['The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.', ' The fire started as two separate fires on November 15, 2008.', ' The "Freeway Fire" started first shortly after 9am with the "Landfill Fire" igniting approximately 2 hours later.', ' These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda.'], ['William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.', ' It is most known for their premium jeans.', ' On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.', ' The label also produces other clothing items such as jackets and tops.', ' The company started first as a denim line, later evolving into a men’s and women’s clothing line.']]}}

>>> datasets['train'][0]['context']['sentences']
[["Radio City is India's first private FM radio station and was started on 3 July 2001.", ' It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).', ' It plays Hindi, English and regional songs.', ' It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.', ' Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.', ' The Radio station currently plays a mix of Hindi and Regional music.', ' Abraham Thomas is the CEO of the company.'], ['Football in Albania existed before the Albanian Football Federation (FSHF) was created.', " This was evidenced by the team's registration at the Balkan Cup tournament during 1929-1931, which started in 1929 (although Albania eventually had pressure from the teams because of competition, competition started first and was strong enough in the duels) .", ' Albanian National Team was founded on June 6, 1930, but Albania had to wait 16 years to play its first international match and then defeated Yugoslavia in 1946.', ' In 1932, Albania joined FIFA (during the 12–16 June convention ) And in 1954 she was one of the founding members of UEFA.'], ['Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.', ' Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.', ' Echosmith started first as "Ready Set Go!"', ' until they signed to Warner Bros.', ' Records in May 2012.', ' They are best known for their hit song "Cool Kids", which reached number 13 on the "Billboard" Hot 100 and was certified double platinum by the RIAA with over 1,200,000 sales in the United States and also double platinum by ARIA in Australia.', ' The song was Warner Bros.', " Records' fifth-biggest-selling-digital song of 2014, with 1.3 million downloads sold.", ' The band\'s debut album, "Talking Dreams", was released on October 8, 2013.'], ["Women's colleges in the Southern United States refers to undergraduate, bachelor's degree–granting institutions, often liberal arts colleges, whose student populations consist exclusively or almost exclusively of women, located in the Southern United States.", " Many started first as girls' seminaries or academies.", ' Salem College is the oldest female educational institution in the South and Wesleyan College is the first that was established specifically as a college for women.', ' Some schools, such as Mary Baldwin University and Salem College, offer coeducational courses at the graduate level.'], ['The First Arthur County Courthouse and Jail, was perhaps the smallest court house in the United States, and serves now as a museum.'], ["Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.", ' Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.', ' In May 1846 it was merged into "Godey\'s Lady\'s Book".'], ['The 2014–15 Ukrainian Hockey Championship was the 23rd season of the Ukrainian Hockey Championship.', ' Only four teams participated in the league this season, because of the instability in Ukraine and that most of the clubs had economical issues.', ' Generals Kiev was the only team that participated in the league the previous season, and the season started first after the year-end of 2014.', ' The regular season included just 12 rounds, where all the teams went to the semifinals.', ' In the final, ATEK Kiev defeated the regular season winner HK Kremenchuk.'], ["First for Women is a woman's magazine published by Bauer Media Group in the USA.", ' The magazine was started in 1989.', ' It is based in Englewood Cliffs, New Jersey.', ' In 2011 the circulation of the magazine was 1,310,696 copies.'], ['The Freeway Complex Fire was a 2008 wildfire in the Santa Ana Canyon area of Orange County, California.', ' The fire started as two separate fires on November 15, 2008.', ' The "Freeway Fire" started first shortly after 9am with the "Landfill Fire" igniting approximately 2 hours later.', ' These two separate fires merged a day later and ultimately destroyed 314 residences in Anaheim Hills and Yorba Linda.'], ['William Rast is an American clothing line founded by Justin Timberlake and Trace Ayala.', ' It is most known for their premium jeans.', ' On October 17, 2006, Justin Timberlake and Trace Ayala put on their first fashion show to launch their new William Rast clothing line.', ' The label also produces other clothing items such as jackets and tops.', ' The company started first as a denim line, later evolving into a men’s and women’s clothing line.']]
"""
# design Dataset that can return a RAGQueryLoader (for inference time) based on Zacks PR
# Zack's PR: https://github.com/pyg-team/pytorch_geometric/pull/9666
# otherwise follow webqsp design for finetuning a G-retriever model but rely on RAG Query Loader

# make sure include flag to support only loading a % of the dataset

class HotPotQA(InMemoryDataset):
    r"""The WebQuestionsSP dataset of the `"The Value of Semantic Parse
    Labeling for Knowledge Base Question Answering"
    <https://aclanthology.org/P16-2033/>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, force_reload=force_reload)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def process(self) -> None:
        import datasets
        import pandas as pd

        datasets = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'sentence-transformers/all-roberta-large-v1'
        model = SentenceTransformer(model_name).to(device)
        model.eval()

        for dataset, path in zip(
            [datasets['train'], datasets['validation'], datasets['test']],
                self.processed_paths,
        ):
            questions = [example["question"] for example in dataset]
            question_embs = model.encode(
                questions,
                batch_size=256,
                output_device='cpu',
            )

            data_list = []
            for i, example in enumerate(tqdm(dataset)):
                raw_nodes: Dict[str, int] = {}
                raw_edges = []
                for tri in example["graph"]:
                    h, r, t = tri
                    h = h.lower()
                    t = t.lower()
                    if h not in raw_nodes:
                        raw_nodes[h] = len(raw_nodes)
                    if t not in raw_nodes:
                        raw_nodes[t] = len(raw_nodes)
                    raw_edges.append({
                        "src": raw_nodes[h],
                        "edge_attr": r,
                        "dst": raw_nodes[t]
                    })
                nodes = pd.DataFrame([{
                    "node_id": v,
                    "node_attr": k,
                } for k, v in raw_nodes.items()],
                                     columns=["node_id", "node_attr"])
                edges = pd.DataFrame(raw_edges,
                                     columns=["src", "edge_attr", "dst"])

                nodes.node_attr = nodes.node_attr.fillna("")
                x = model.encode(
                    nodes.node_attr.tolist(),
                    batch_size=256,
                    output_device='cpu',
                )
                edge_attr = model.encode(
                    edges.edge_attr.tolist(),
                    batch_size=256,
                    output_device='cpu',
                )
                edge_index = torch.tensor([
                    edges.src.tolist(),
                    edges.dst.tolist(),
                ], dtype=torch.long)

                question = f"Question: {example['question']}\nAnswer: "
                label = ('|').join(example['answer']).lower()
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
                data, desc = retrieval_via_pcst(
                    data,
                    question_embs[i],
                    nodes,
                    edges,
                    topk=3,
                    topk_e=5,
                    cost_e=0.5,
                )
                data.question = question
                data.label = label
                data.desc = desc
                data_list.append(data)

            self.save(data_list, path)
