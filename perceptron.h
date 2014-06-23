#include "stdafx.h"
#include "data.h"
#include "decoder.h"

struct Thread_data
{

	vector<vector<int> > *cur_line_ptr;
	Model *model_ptr;
	vector<int> *output_ptr;
};

class Perceptron
{
	public:
		Perceptron(Data *data,Model *model,size_t line,size_t round);
		void train(string &train_file);
		void test(string &test_file);

	private:
		size_t ROUND;
		size_t LINE;
		size_t m_line;
		size_t m_round;
		Model *m_model;
		Data *m_data;
};
