""" Contains code for encoding an input sequence. """

import torch
import torch.nn.functional as F
from .torch_utils import create_multilayer_lstm_params, encode_sequence,encode_sequence_bert

from .gated_graph_conv import GatedGraphConv

class Encoder(torch.nn.Module):
    """ Encodes an input sequence. """
    def __init__(self, num_layers, input_size, state_size):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-ef")
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-eb")

    def forward(self, sequence, embedder, dropout_amount=0.):
        """ Encodes a sequence forward and backward.
        Inputs:
            forward_seq (list of str): The string forwards.
            backward_seq (list of str): The string backwards.
            f_rnns (list of dy.RNNBuilder): The forward RNNs.
            b_rnns (list of dy.RNNBuilder): The backward RNNS.
            emb_fn (dict str->dy.Expression): Embedding function for tokens in the
                sequence.
            size (int): The size of the RNNs.
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        forward_state, forward_outputs = encode_sequence(
            sequence,
            self.forward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        backward_state, backward_outputs = encode_sequence(
            sequence[::-1],
            self.backward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        cell_memories = []
        hidden_states = []
        for i in range(self.num_layers):
            cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
            hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

        assert len(forward_outputs) == len(backward_outputs)

        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(sequence)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs



#
class SchemaEncoder1(torch.nn.Module):
    """ 
    Encodes an input sequence with  
    #TODO: graph encoding 
    """
    def __init__(self, num_layers, input_size, state_size):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-ef")
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-eb")

    def forward(self, sequence, embedder, dropout_amount=0.):
        """ Encodes a sequence forward and backward.
        Inputs:
            forward_seq (list of str): The string forwards.
            backward_seq (list of str): The string backwards.
            f_rnns (list of dy.RNNBuilder): The forward RNNs.
            b_rnns (list of dy.RNNBuilder): The backward RNNS.
            emb_fn (dict str->dy.Expression): Embedding function for tokens in the
                sequence.
            size (int): The size of the RNNs.
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        forward_state, forward_outputs = encode_sequence(
            sequence,
            self.forward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        backward_state, backward_outputs = encode_sequence(
            sequence[::-1],
            self.backward_lstms,
            embedder,
            dropout_amount=dropout_amount)

        cell_memories = []
        hidden_states = []
        for i in range(self.num_layers):
            cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
            hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

        assert len(forward_outputs) == len(backward_outputs)

        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(sequence)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs

class Encoder_Gnn(torch.nn.Module):
    """ Encodes an input sequence. """
    def __init__(self, num_layers, input_size, state_size):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-ef")
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-eb")

        self.l1 = torch.nn.Linear(768,int(input_size))
        
            

    def forward(self, last_hidden, dropout_amount=0.):
        """ Encodes a sequence forward and backward. 

	10/12 - Add Bert Utterance embedding

        Inputs:
            last_hidden (hidden states from bert): 
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        # print(sequence, len(sequence))
        # bert utterance encoding
        forward_state = None
        forward_outputs = None
        backward_state = None
        backward_outputs = None
        cell_memories = []
        hidden_states = []
        
        last_hidden = [last_hidden[:,i,:].squeeze() for i in range(last_hidden.size()[1])] # size [batch=1, q_len, hidden ]

            
    
        forward_state, forward_outputs = encode_sequence_bert(
            last_hidden,
            self.forward_lstms,
            dropout_amount=dropout_amount)
        # print(forward_state[0][0].size(),forward_state[1][0].size())
        backward_state, backward_outputs = encode_sequence_bert(
            last_hidden[::-1],
            self.backward_lstms,
            dropout_amount=dropout_amount)

        # cell_memories = []
        # hidden_states = []
        for i in range(self.num_layers):
            cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
            hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

        assert len(forward_outputs) == len(backward_outputs)

        
        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(last_hidden)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs

class Encoder_Bert(torch.nn.Module):
    """ Encodes an input sequence. """
    def __init__(self, num_layers, input_size, state_size, from_pretrained=False, pretrained_weights='bert-base-uncased'):
        super().__init__()

        self.num_layers = num_layers
        self.forward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-ef")
        self.backward_lstms = create_multilayer_lstm_params(self.num_layers, input_size, state_size / 2, "LSTM-eb")
        self.use_bert =  from_pretrained
        self.l1 = torch.nn.Linear(768,int(input_size))
        
        if from_pretrained:
            print('From pretrained')
            self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
            self.bert_model = BertModel.from_pretrained(pretrained_weights)
            

    def forward(self, sequence, embedder, dropout_amount=0.):
        """ Encodes a sequence forward and backward. 

	10/12 - Add Bert Utterance embedding

        Inputs:
            forward_seq (list of str): The string forwards.
            backward_seq (list of str): The string backwards.
            f_rnns (list of dy.RNNBuilder): The forward RNNs.
            b_rnns (list of dy.RNNBuilder): The backward RNNS.
            emb_fn (dict str->dy.Expression): Embedding function for tokens in the
                sequence.
            size (int): The size of the RNNs.
            dropout_amount (float, optional): The amount of dropout to apply.

        Returns:
            (list of dy.Expression, list of dy.Expression), list of dy.Expression,
            where the first pair is the (final cell memories, final cell states) of
            all layers, and the second list is a list of the final layer's cell
            state for all tokens in the sequence.
        """
        # print(sequence, len(sequence))
        # bert utterance encoding
        forward_state = None
        forward_outputs = None
        backward_state = None
        backward_outputs = None
        cell_memories = []
        hidden_states = []
        if self.use_bert:
            input_ids = torch.tensor([self.bert_tokenizer.encode(sequence, add_special_tokens=True)],device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            last_hidden = None
        
            with torch.no_grad():
	        # create a list of tensor embedding corresponding to words in sequence
                last_hidden = self.bert_model(input_ids)[0]
                # print(last_hidden[:,0,:].size())

            last_hidden = self.l1(last_hidden)
            last_hidden = [last_hidden[:,i,:].squeeze() for i in range(last_hidden.size()[1])] # size [batch=1, q_len, hidden ]

                
            
        
            forward_state, forward_outputs = encode_sequence_bert(
                last_hidden,
                self.forward_lstms,
                dropout_amount=dropout_amount)
            # print(forward_state[0][0].size(),forward_state[1][0].size())
            backward_state, backward_outputs = encode_sequence_bert(
                last_hidden[::-1],
                self.backward_lstms,
                dropout_amount=dropout_amount)

            # cell_memories = []
            # hidden_states = []
            for i in range(self.num_layers):
                cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
                hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

            assert len(forward_outputs) == len(backward_outputs)

        else: 
            forward_state, forward_outputs = encode_sequence(
                sequence,
                self.forward_lstms,
                embedder,
                dropout_amount=dropout_amount)
            # print(forward_state[0][0].size(),forward_state[1][0].size())
            backward_state, backward_outputs = encode_sequence(
                sequence[::-1],
                self.backward_lstms,
                embedder,
                dropout_amount=dropout_amount)

            # cell_memories = []
            # hidden_states = []
            for i in range(self.num_layers):
                cell_memories.append(torch.cat([forward_state[0][i], backward_state[0][i]], dim=0))
                hidden_states.append(torch.cat([forward_state[1][i], backward_state[1][i]], dim=0))

            assert len(forward_outputs) == len(backward_outputs)
        

        backward_outputs = backward_outputs[::-1]

        final_outputs = []
        for i in range(len(sequence)):
            final_outputs.append(torch.cat([forward_outputs[i], backward_outputs[i]], dim=0))

        return (cell_memories, hidden_states), final_outputs