""" Contains the class for an interaction in ATIS. """

from . import anonymization as anon
from . import sql_util
from .snippets import expand_snippets
from .utterance import Utterance, OUTPUT_KEY, ANON_INPUT_KEY

import torch

class Schema:
    def __init__(self, table_schema, simple=False,gnn=True):
        if simple:
            self.helper1(table_schema)
        else:
            self.helper2(table_schema)
        # if gnn:
        #     self.prepare_input_gnn()
        #     print(self.nodes)

    def helper1(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index.append(i)

        column_keep_index_2 = []
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = table_name.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index_2.append(i)

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name
            if i in column_keep_index_2:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        max_id_1 = max(v for k,v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface_form)

    def helper2(self, table_schema):
        self.table_schema = table_schema
        column_names = table_schema['column_names']
        column_names_original = table_schema['column_names_original']
        table_names = table_schema['table_names']
        table_names_original = table_schema['table_names_original']
        assert len(column_names) == len(column_names_original) and len(table_names) == len(table_names_original)

        column_keep_index = []

        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names_original):
            if table_id >= 0:
                table_name = table_names_original[table_id]
                column_name_surface_form = '{}.{}'.format(table_name,column_name)
            else:
                column_name_surface_form = column_name
            column_name_surface_form = column_name_surface_form.lower()
            if column_name_surface_form not in self.column_names_surface_form_to_id:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
                column_keep_index.append(i)

        start_i = len(self.column_names_surface_form_to_id)
        for i, table_name in enumerate(table_names_original):
            column_name_surface_form = '{}.*'.format(table_name.lower())
            self.column_names_surface_form.append(column_name_surface_form)
            self.column_names_surface_form_to_id[column_name_surface_form] = i + start_i

        self.column_names_embedder_input = []
        self.column_names_embedder_input_to_id = {}
        for i, (table_id, column_name) in enumerate(column_names):
            if table_id >= 0:
                table_name = table_names[table_id]
                column_name_embedder_input = table_name + ' . ' + column_name
            else:
                column_name_embedder_input = column_name
            if i in column_keep_index:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1

        start_i = len(self.column_names_embedder_input_to_id)
        for i, table_name in enumerate(table_names):
            column_name_embedder_input = table_name + ' . *'
            self.column_names_embedder_input.append(column_name_embedder_input)
            self.column_names_embedder_input_to_id[column_name_embedder_input] = i + start_i

        assert len(self.column_names_surface_form) == len(self.column_names_surface_form_to_id) == len(self.column_names_embedder_input) == len(self.column_names_embedder_input_to_id)

        max_id_1 = max(v for k,v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1

        self.num_col = len(self.column_names_surface_form)

    def prepare_input_gnn(self,pad_len=12):
        """
        Return: Nodes(list of tokenized db items)
        Return: masks 
        Return: relations(lists of list of related columns), inner list corresponds to edge type
        """

        all_hds = self.column_names_embedder_input      # table name . column

        tables = []
        tb_name = {} # index of header in node - len(nodes)
        columns = {}
        nodes = []
        relations = [[],[],[]] # three edge types, we use tb_name.col as embedding 
        # print(relations)
        all_columns = {}

        # print(1111111,nlu_t1,all_hds)

        nodes.append('*')
        for i in all_hds:
            # print(i.split('.'))
            if i != "*" and len(i.split('.')) > 1:
                
                header,col  = i.split('.')
                # if col.strip() != '*':
                # print(header,col)
                # first add headers
                nodes.append(i)
                # if not col in columns:
                if not header in tables:
                    tables.append(header) 
                    tb_name[header] = len(tables) -1
                    #columns[col]= len(nodes)-1 # add column name to columns with index in nodes as value

                # take redundancy for foreign key
                if col in columns: # find('id') != -1
                    # print('key')
                    relations[2].append([tb_name[header],columns[col]])  # add foreign key relation
                else:
                    # column id
                    columns[col] = len(nodes) -1
                    
                    # assume primary key have "id"
                    if col.find("id") != -1:
                        # print('primary')
                        relations[1].append([tb_name[header],columns[col]])
                    else:
                        relations[0].append([tb_name[header],columns[col]])
    # for *

        # nodes += tables
        base = len(nodes)
        nodes += tables
        for i in relations:
            for j in i:
                j[0] += base
        # tokenize nodes to feed into model
        
            # print(nodes[i],masks[i])

        self.nodes = nodes
        self.relations =  relations
        # self.masks = masks
        # self.new_schema = new_schema
        self.num_col = len(self.nodes)


    def __len__(self):
        return self.num_col

    def in_vocabulary(self, column_name, surface_form=False):
        # print(column_name)
        if surface_form:
            print(self.column_names_surface_form_to_id)
            return column_name in self.column_names_surface_form_to_id
        else:
            return column_name in self.column_names_embedder_input_to_id

    def column_name_embedder_bow(self, column_name, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
            column_name_embedder_input = self.column_names_embedder_input[column_name_id]
        else:
            column_name_embedder_input = column_name

        column_name_embeddings = [column_name_token_embedder(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)

    def set_column_name_embeddings(self, column_name_embeddings, column_names=None):
        # print(self.column_names_embedder_input_to_id,self.column_names_surface_form_to_id)
        if column_names and len(column_names[1]) > 1 :
            # print(1231213123,column_names)
            self.column_names_surface_form = column_names
            
            # surface form is not quite correct
            self.column_names_embedder_input = [i.replace(" . ", ".").replace(" ","_") for i in column_names]
            self.column_names_surface_form = [i.replace(" . ", ".").replace(" ","_").replace("."," . ") for i in column_names]
            for i,item in enumerate(self.column_names_surface_form):
                self.column_names_surface_form_to_id[item] = i
            for i,item in enumerate(self.column_names_embedder_input):
                self.column_names_embedder_input_to_id[item] = i
            # print(self.column_names_embedder_input)
            # print(self.column_names_surface_form)
        #     len1 = len(self.column_names_surface_form)
        #     # self.column_names_surface_form_to_id = {}
            # self.column_names_embedder_input_to_id = {}
            # for i in range(len(column_names)):
            #     self.column_names_surface_form_to_id[column_names[i]] = i + len1 
            
            #     self.column_names_embedder_input_to_id[column_names[i]] = i + len1

            # self.column_names_surface_form += column_names
            # self.column_names_embedder_input += column_names
            self.num_col = len(column_names)

        # self.column_name_embeddings = [torch.zeros(column_name_embeddings[0].size(0)).to(device)] * len1 + column_name_embeddings
        self.column_name_embeddings = column_name_embeddings
        # print(len(self.column_name_embeddings),self.num_col)
        assert len(self.column_name_embeddings) == self.num_col

    def column_name_embedder(self, column_name, surface_form=False):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
        else:
            column_name_id = self.column_names_embedder_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]

class Interaction:
    """ ATIS interaction class.

    Attributes:
        utterances (list of Utterance): The utterances in the interaction.
        snippets (list of Snippet): The snippets that appear through the interaction.
        anon_tok_to_ent:
        identifier (str): Unique identifier for the interaction in the dataset.
    """
    def __init__(self,
                 utterances,
                 schema,
                 snippets,
                 anon_tok_to_ent,
                 identifier,
                 params):
        self.utterances = utterances
        self.schema = schema
        self.snippets = snippets
        self.anon_tok_to_ent = anon_tok_to_ent
        self.identifier = identifier

        # Ensure that each utterance's input and output sequences, when remapped
        # without anonymization or snippets, are the same as the original
        # version.
        for i, utterance in enumerate(self.utterances):
            deanon_input = self.deanonymize(utterance.input_seq_to_use,
                                            ANON_INPUT_KEY)
            assert deanon_input == utterance.original_input_seq, "Anonymized sequence [" \
                + " ".join(utterance.input_seq_to_use) + "] is not the same as [" \
                + " ".join(utterance.original_input_seq) + "] when deanonymized (is [" \
                + " ".join(deanon_input) + "] instead)"
            desnippet_gold = self.expand_snippets(utterance.gold_query_to_use)
            deanon_gold = self.deanonymize(desnippet_gold, OUTPUT_KEY)
            assert deanon_gold == utterance.original_gold_query, \
                "Anonymized and/or snippet'd query " \
                + " ".join(utterance.gold_query_to_use) + " is not the same as " \
                + " ".join(utterance.original_gold_query)

    def __str__(self):
        string = "Utterances:\n"
        for utterance in self.utterances:
            string += str(utterance) + "\n"
        string += "Anonymization dictionary:\n"
        for ent_tok, deanon in self.anon_tok_to_ent.items():
            string += ent_tok + "\t" + str(deanon) + "\n"

        return string

    def __len__(self):
        return len(self.utterances)

    def deanonymize(self, sequence, key):
        """ Deanonymizes a predicted query or an input utterance.

        Inputs:
            sequence (list of str): The sequence to deanonymize.
            key (str): The key in the anonymization table, e.g. NL or SQL.
        """
        return anon.deanonymize(sequence, self.anon_tok_to_ent, key)

    def expand_snippets(self, sequence):
        """ Expands snippets for a sequence.

        Inputs:
            sequence (list of str): A SQL query.

        """
        return expand_snippets(sequence, self.snippets)

    def input_seqs(self):
        in_seqs = []
        for utterance in self.utterances:
            in_seqs.append(utterance.input_seq_to_use)
        return in_seqs

    def output_seqs(self):
        out_seqs = []
        for utterance in self.utterances:
            out_seqs.append(utterance.gold_query_to_use)
        return out_seqs

def load_function(parameters,
                  nl_to_sql_dict,
                  anonymizer,
                  database_schema=None):
    def fn(interaction_example):
        keep = False

        raw_utterances = interaction_example["interaction"]

        if "database_id" in interaction_example:
            database_id = interaction_example["database_id"]
            interaction_id = interaction_example["interaction_id"]
            identifier = database_id + '/' + str(interaction_id)
        else:
            identifier = interaction_example["id"]

        schema = None
        if database_schema:
            if 'removefrom' not in parameters.data_directory:
                schema = Schema(database_schema[database_id], simple=True)
            else:
                schema = Schema(database_schema[database_id])

        snippet_bank = []

        utterance_examples = []

        anon_tok_to_ent = {}

        for utterance in raw_utterances:
            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1]

            proc_utterance = Utterance(utterance,
                                       available_snippets,
                                       nl_to_sql_dict,
                                       parameters,
                                       anon_tok_to_ent,
                                       anonymizer)
            keep_utterance = proc_utterance.keep

            if schema:
                assert keep_utterance

            if keep_utterance:
                keep = True
                utterance_examples.append(proc_utterance)

                # Update the snippet bank, and age each snippet in it.
                if parameters.use_snippets:
                    if 'atis' in parameters.data_directory:
                        snippets = sql_util.get_subtrees(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)
                    else:
                        snippets = sql_util.get_subtrees_simple(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)

                    for snippet in snippets:
                        snippet.assign_id(len(snippet_bank))
                        snippet_bank.append(snippet)

                for snippet in snippet_bank:
                    snippet.increase_age()

        interaction = Interaction(utterance_examples,
                                  schema,
                                  snippet_bank,
                                  anon_tok_to_ent,
                                  identifier,
                                  parameters)

        return interaction, keep

    return fn
