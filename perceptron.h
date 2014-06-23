#include "stdafx.h"
#include "data.h"
#include "decoder.h"

struct Thread_data
{

	vector<vector<int> > *cur_line_ptr;
	Model *model_ptr;
	vector<int> *output_ptr;
	vector<int> *gold_ptr;
	bool mode;
};

class Perceptron
{
	public:
		Perceptron(Data *data,Model *model,size_t line,size_t round, size_t core);
		void train(string &train_file);
		void test(string &test_file);

	private:
		size_t ROUND;
		size_t LINE;
		size_t m_line;
		size_t m_round;
		size_t NUM_THREADS;
		Model *m_model;
		Data *m_data;
};
