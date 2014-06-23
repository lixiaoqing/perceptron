#include "perceptron.h"

void *decode_thread(void *thread_arg)
{
	Thread_data *my_thread_data;
	my_thread_data = (Thread_data *) thread_arg;
	Decoder my_decoder(my_thread_data->cur_line_ptr,my_thread_data->model_ptr);
	vector<int> output_taglist = my_decoder.decode();
	my_thread_data->output_ptr->clear();
	for (size_t i=0;i<output_taglist.size();i++)
	{
		my_thread_data->output_ptr->push_back(output_taglist.at(i));
	}
	pthread_exit(NULL);
}

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
				size_t end_pos = taglist_output.size();
				for (size_t i=2;i<end_pos;i++)
				{
					vector<vector<int> > local_features = m_decoder.extract_features(taglist_output,i);
					vector<vector<int> > local_gold_features = m_decoder.extract_features(taglist_gold,i);
					m_model->update_paras(local_features,local_gold_features,m_round,m_line);
				}
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
	/*
	*/
	const size_t num_threads = 4;
	size_t num_rounds = LINE/num_threads;
	for (size_t r=0;r<num_rounds;r++)
	{
		pthread_t threads[num_threads];
		Thread_data tdata[num_threads];
		for (size_t i=0;i<num_threads;i++)
		{
			size_t line = r*num_threads + i;
			vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(line);
			vector<int> *output_ptr = &outputs.at(line);
			tdata[i] = {cur_line_ptr,m_model,output_ptr};
		}
		for (size_t i=0;i<num_threads;i++)
		{
			pthread_create(&threads[i], NULL, decode_thread, (void *)&tdata[i]);
		}
		for (size_t i=0;i<num_threads;i++)
		{
			pthread_join(threads[i],NULL);
		}
	}

	for(size_t m_line=num_threads*num_rounds;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		Decoder m_decoder(cur_line_ptr,m_model);
		outputs.at(m_line) = m_decoder.decode();
	}

/*
//#pragma omp parallel for
	for(size_t m_line=0;m_line<LINE;m_line++)
	{
		vector<vector<int> > *cur_line_ptr = m_data->get_token_matrix_ptr(m_line);
		Decoder m_decoder(cur_line_ptr,m_model);
		outputs.at(m_line) = m_decoder.decode();
	}
*/

	for(size_t m_line=0;m_line<LINE;m_line++)
	{
		//cout<<"output line "<<m_line<<endl;
		//cout<<"output size "<<outputs.at(m_line).size()<<endl;
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
		cout<<"usage:\n./pcpt -t training_file -m model_file -i iter_num\n./pcpt -d testing_file -m model_file\n";
		return 0;
	}

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
	cout<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC<<endl;
	return 0;
}
