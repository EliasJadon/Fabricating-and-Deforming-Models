#pragma once

class UniqueColors {
private:
	std::vector<Eigen::Vector3f> colors;
	int index;
	float add(const float x, const float num) {
		float r = x * 255.0f;
		r += num;
		while (r < 0) {
			r += 255;
		}
		r = (int)r % 255;
		return r / 255.0f;
	}
public:
	UniqueColors() {
		auto put_color = [&](const float r, const float g, const float b) {
			this->colors.push_back(Eigen::Vector3f(r / 255.0f, g / 255.0f, b / 255.0f));
		};
		index = 0;
		put_color(255, 0, 0); //red
		put_color(0, 255, 0); //Lime
		put_color(0, 0, 255); //Blue
		put_color(255, 255, 0); //Yellow
		put_color(0, 255, 255); //Cyan / Aqua
		put_color(255, 0, 255); //Magenta / Fuchsia
		put_color(192, 192, 192); //Silver
		put_color(128, 128, 128); //Gray
		put_color(128, 0, 0); //Maroon
		put_color(128, 128, 0); //Olive
		put_color(0, 128, 0); //Green
		put_color(128, 0, 128); //Purple
		put_color(0, 128, 128); //Teal
		put_color(0, 0, 128); //Navy
		put_color(178, 34, 34); //firebrick
		put_color(255, 165, 0); //orange
		put_color(184, 134, 11); //dark golden rod
		put_color(218, 165, 32); //golden rod
		put_color(0, 233, 154); //medium spring green
		put_color(102, 205, 170); //medium aqua marine
		put_color(95, 158, 160); //cadet blue
		put_color(221, 160, 221); //plum
		put_color(218, 112, 214); //orchid
		put_color(245, 222, 179); //wheat
		put_color(205, 133, 63); //peru
		put_color(210, 105, 30); //chocolate
		put_color(230, 230, 250); //lavender
		put_color(240, 248, 255); //alice blue
		put_color(0, 0, 0); //black
		put_color(220, 220, 220); //gainsboro
	}
	Eigen::Vector3f getNext()
	{
		Eigen::Vector3f c = colors[index];
		colors[index] << add(c(0), 18), add(c(1), -18), add(c(2), 60);
		if ((++index) >= colors.size())
			index = 0;
		return c;
	}
};

class ColorsHashMap {
private:
	double MinDistance;
	std::vector<Eigen::Vector3d>* colors;
	std::vector<Eigen::Vector3d> points;
	double getRand() {
		//random number from 0 to 1
		return (double)rand() / RAND_MAX;
	}
public:
	ColorsHashMap(const double MinDistance, std::vector<Eigen::Vector3d>* colors) {
		this->MinDistance = MinDistance;
		this->colors = colors;
		points.clear();
	}
	Eigen::Vector3d getColor(Eigen::Vector3d point) {
		//Check if the point exists
		int argmin = -1;
		double min = 99999;
		for (int i = 0; i < points.size(); i++) {
			double currDistance = (points[i] - point).squaredNorm();
			if ((currDistance < MinDistance) && (currDistance < min)) {
				argmin = i;
				min = currDistance;
			}
		}
		if (argmin != -1)
			return (*colors)[argmin];

		//Then, add a new point
		points.push_back(point);
		if (colors->size() < points.size())
			colors->push_back(Eigen::Vector3d(getRand(), getRand(), getRand()));
		return (*colors)[points.size() - 1];
	}
};
