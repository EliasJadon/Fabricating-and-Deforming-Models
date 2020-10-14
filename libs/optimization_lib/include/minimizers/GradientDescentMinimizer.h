#pragma once

#include "libs/optimization_lib/include/minimizers/Minimizer.h"

class GradientDescentMinimizer : public Minimizer
{
public:
	GradientDescentMinimizer(const int solverID) : Minimizer(solverID) {}
	virtual void step() override {
		objective->updateX(X);
		currentEnergy = objective->value(true);
		objective->gradient(g, true);
		p = -g;
	}
	virtual void internal_init() override {
		objective->updateX(X);
		g.resize(X.size());
	}
};