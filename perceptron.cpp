#include "perceptron.h"

void *decode_thread(void *thread_arg)
{
	Thread_data *my_thread_data;
	my_thread_data = (Thread_data *) thread_arg;
	Decoder my_decoder(my_thread_data->cur_line_ptr,my_thread_data->model_ptr,my_thread_data->mode);
	vector<int> output;
	output = my_decoder.decode();
	my_thread_data->output_ptr->clear();
	for (size_t i=0;i<output.size();i++)
	{
		my_thread_data->output_ptr->push_back(output.at(i));
	}
	if (my_thread_data->mode == true)
	{
		vector<int> gold = my_decoder.m_gold_taglist;
		my_thread_data->gold_ptr->clear();
		for (size_t i=0;i<output.size();i++)
		{
			my_thread_data->gold_ptr->push_back(gold.at(i));
		}
	}
	pthread_exit(NULL);
}

Perceptron::Perceptron(Data *data,Model *model,size_t line,size_t round,size_t core)
{
	ROUND = round;
	LINE = line;
	m_line = 0;
	m_round = 0;
	NUM_THREADS = core;
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

		size_t para_round = LINE/NUM_THREADS;
		for (size_t r=0;r<para_round;r++)
		{
			vector<vector<int> > outputs;
			outputs.resize(NUM_THREADS);
			vector<vector<int> > golds;
			golds.resize(NUM_THREADS);
			vector<pthread_t> threads;
			threads.resize(NUM_THREADS);
			vector<Thread_data> tdata;
			tdata.resize(NUM_THREADS);
			for (size_t i=0;i<NUM_THREADS;i++)
			{
				m_line = r*NUM_THREADS + i;
				vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
				vector<int> *output_ptr = &outputs.at(i);
				vector<int> *gold_ptr = &golds.at(i);
				tdata[i] = {cur_line_ptr,m_model,output_ptr,gold_ptr,true};
			}
			for (size_t i=0;i<NUM_THREADS;i++)
			{
				pthread_create(&threads[i], NULL, decode_thread, (void *)&tdata[i]);
			}
			for (size_t i=0;i<NUM_THREADS;i++)
			{
				pthread_join(threads[i],NULL);
			}
			for (size_t i=0;i<NUM_THREADS;i++)
			{
				m_line = r*NUM_THREADS + i;
				if (m_line%sect_size == 0)
				{
					cout<<'.';
					cout.flush();
				}
				vector<int> &output = outputs.at(i);
				vector<int> &gold = golds.at(i);
				if (output == gold)
				{
					continue;
				}
				size_t end_pos = output.size();
				for (size_t j=2;j<end_pos;j++)
				{
					vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
					Decoder m_decoder(cur_line_ptr,m_model,true);
					vector<vector<int> > local_features = m_decoder.extract_features(output,j);
					vector<vector<int> > local_gold_features = m_decoder.extract_features(gold,j);
					m_model->update_paras(local_features,local_gold_features,m_round,m_line);
				}
			}
		}

		for(size_t m_line=NUM_THREADS*para_round;m_line<LINE;m_line++)
		{
			vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
			Decoder m_decoder(cur_line_ptr,m_model,true);
			vector<int> output = m_decoder.decode();
			vector<int> gold = m_decoder.m_gold_taglist;

			if (output == gold)
			{
				continue;
			}
			size_t end_pos = output.size();
			for (size_t j=2;j<end_pos;j++)
			{
				vector<vector<int> > local_features = m_decoder.extract_features(output,j);
				vector<vector<int> > local_gold_features = m_decoder.extract_features(gold,j);
				m_model->update_paras(local_features,local_gold_features,m_round,m_line);
			}
		}

		if (m_line == LINE - 1)
		{
			m_model->update_paras_for_lastline(m_round,m_line);
		}

		cout<<"\t iter"<<m_round<<endl;
		m_model->save_bin_model(m_round);
	}
	//m_model->save_bin_model(m_round);
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
	size_t num_rounds = LINE/NUM_THREADS;
	for (size_t r=0;r<num_rounds;r++)
	{
		vector<pthread_t> threads;
		threads.resize(NUM_THREADS);
		vector<Thread_data> tdata;
		tdata.resize(NUM_THREADS);
		for (size_t i=0;i<NUM_THREADS;i++)
		{
			size_t line = r*NUM_THREADS + i;
			vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(line);
			vector<int> *output_ptr = &outputs.at(line);
			tdata[i] = {cur_line_ptr,m_model,output_ptr,NULL,false};
		}
		for (size_t i=0;i<NUM_THREADS;i++)
		{
			pthread_create(&threads[i], NULL, decode_thread, (void *)&tdata[i]);
		}
		for (size_t i=0;i<NUM_THREADS;i++)
		{
			pthread_join(threads[i],NULL);
		}
	}

	for(size_t m_line=NUM_THREADS*num_rounds;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		Decoder m_decoder(cur_line_ptr,m_model,false);
		outputs.at(m_line) = m_decoder.decode();
	}

	for(size_t m_line=0;m_line<LINE;m_line++)
	{
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

	if (argc == 1)
	{
		cout<<"usage:\n./pcpt -t training_file -m model_file -i iter_num -c core\n./pcpt -d testing_file -m model_file -c core\n";
		return 0;
	}

	if (argv[1][1] == 't')
	{
		string train_file(argv[2]);
		string model_file(argv[4]);
		size_t round(stoi(argv[6]));
		size_t core(stoi(argv[8]));
		Data my_data(true,train_file);
		size_t line = my_data.get_size();
		Model my_model(true,model_file,line,round);
		Perceptron my_pcpt(&my_data,&my_model,line,round,core);
		my_pcpt.train(train_file);
	}
	else if (argv[1][1] == 'd')
	{
		string test_file(argv[2]);
		string model_file(argv[4]);
		size_t core(stoi(argv[6]));
		Data my_data(false,test_file);
		size_t line = my_data.get_size();
		Model my_model(false,model_file,line,0);
		Perceptron my_pcpt(&my_data,&my_model,line,0,core);
		my_pcpt.test(test_file);
	}

	b = clock();
	cout<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC<<endl;
	return 0;
}
