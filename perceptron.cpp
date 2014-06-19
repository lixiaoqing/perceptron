#include "perceptron.h"

Perceptron::Perceptron(Data *data,Model *model)
{
	ROUND = 20;
	LINE = 1;
	m_line = 0;
	m_round = 0;
	m_data = data;
	m_model = model;
}

void Perceptron::train(string &train_file)
{
	LINE = m_data->get_size();
	size_t sect_size = LINE/20;
	m_model->set_line(LINE);
	for (m_round=0;m_round<ROUND;m_round++)
	{
		m_data->shuffle();
		for (m_line=0;m_line<LINE;m_line++)
		{
			vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
			Decoder m_decoder(cur_line_ptr,m_model);
			vector<int> taglist_output;
			vector<int> taglist_gold;
			if(m_decoder.decode_for_train(taglist_output,taglist_gold) == false)
			{
				m_model->update_paras(taglist_output,taglist_gold,m_round,m_line);
			}
			if (m_line == LINE - 1)
			{
				m_model->update_paras_for_lastline(m_round,m_line);
			}
			if (m_line%sect_size == 0)
			{
				cout<<'.';
				cout.flush();
			}
		}
		cout<<"\t iter"<<m_round<<endl;
		m_model->save_bin_model(m_round,m_round*LINE);
	}
	//save_model(ROUND*LINE);
}

void Perceptron::test(string &test_file)
{
	LINE = m_data->get_size();
	ofstream fout;
	fout.open("output");
	if (!fout.is_open())
	{
		cerr<<"fail to open output file\n";
		return;
	}

	vector<vector<int> > outputs;
	outputs.resize(LINE);
	/*
	size_t num_threads = 10;
	size_t num_rounds = LINE/num_threads;
	for (size_t r=0;r<num_rounds;r++)
	{
#pragma omp parallel for
		for (size_t i=0;i<num_threads;i++)
		{
			size_t m_line = r*num_threads + i;
			vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
			Decoder m_decoder(cur_line_ptr,m_model);
			outputs.at(m_line) = m_decoder.decode();
		}
	}

	for(size_t m_line=num_threads*num_rounds;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		Decoder m_decoder(cur_line_ptr,m_model);
		outputs.at(m_line) = m_decoder.decode();
	}
	*/
//#pragma omp parallel for
	for(size_t m_line=0;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		Decoder m_decoder(cur_line_ptr,m_model);
		outputs.at(m_line) = m_decoder.decode();
	}

	for(size_t m_line=0;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		for (size_t i=2;i<outputs.at(m_line).size();i++)
		{
			fout<<m_data->m_data_matrix.at(m_line).at(i)<<'\t'<<m_data->get_tag(outputs.at(m_line).at(i))<<endl;
		}
		fout<<endl;
	}
}

int main(int argc, char *argv[])
{
	clock_t a,b;
	a = clock();

	if (argv[1][1] == 't')
	{
		string train_file(argv[2]);
		Data my_data;
		my_data.set_mode("train");
		my_data.load_data(train_file);
		Model my_model;
		my_model.set_mode("train");
		Perceptron my_pcpt(&my_data,&my_model);
		my_pcpt.train(train_file);
	}
	else if (argv[1][1] == 'd')
	{
		Data my_data;
		string test_file(argv[2]);
		my_data.set_mode("test");
		my_data.load_data(test_file);
		Model my_model;
		my_model.load_bin_model();
		my_model.set_mode("test");
		Perceptron my_pcpt(&my_data,&my_model);
		my_pcpt.test(test_file);
	}

	b = clock();
	cout<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC;
	return 0;
}
