#include "perceptron.h"

Perceptron::Perceptron(Data *data,Model *model,size_t line,size_t round)
{
	ROUND = round;
	LINE = line;
	m_line = 0;
	m_round = 0;
	m_data = data;
	m_model = model;
}

void Perceptron::train(string &train_file)
{
	LINE = m_data->get_size();
	size_t sect_size = LINE/20;
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
		//m_model->save_bin_model(m_round);
	}
	m_model->save_bin_model(m_round);
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
		string model_file(argv[4]);
		size_t round(stoi(argv[6]));
		string mode("train");
		Data my_data(mode,train_file);
		size_t line = my_data.get_size();
		Model my_model(mode,model_file,line,round);
		Perceptron my_pcpt(&my_data,&my_model,line,round);
		my_pcpt.train(train_file);
	}
	else if (argv[1][1] == 'd')
	{
		string test_file(argv[2]);
		string model_file(argv[4]);
		string mode("test");
		Data my_data(mode,test_file);
		size_t line = my_data.get_size();
		Model my_model(mode,model_file,line,0);
		Perceptron my_pcpt(&my_data,&my_model,line,0);
		my_pcpt.test(test_file);
	}

	b = clock();
	cout<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC;
	return 0;
}
