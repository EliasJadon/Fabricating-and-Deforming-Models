#pragma once

class FacesGroup {
public:
	Eigen::Vector3f color;
	std::string name;
	std::set<int> faces;

	FacesGroup(const int index) {
		auto put_color = [&](const float r, const float g, const float b) {
			this->color << r / 255.0f, g / 255.0f, b / 255.0f;
		};
		faces.clear();
		if (index == 0)
		{
			put_color(255, 255, 0);//Yellow
			name = "Yellow";
		}
		else if (index == 1)
		{
			put_color(0, 0, 255); //Blue
			name = "Blue";
		}
		else if (index == 2)
		{
			put_color(0, 255, 0); //Green
			name = "Green";
		}
		else if (index == 3)
		{
			put_color(255, 0, 0); //red
			name = "Red";
		}
		else if (index == 4)
		{
			put_color(128, 128, 128); //Gray
			name = "Gray";
		}
		else if (index == 5)
		{
			put_color(128, 0, 128); //Purple
			name = "Purple";
		}
		else if (index == 6)
		{
			put_color(255, 165, 0); //orange
			name = "Orange";
		}
		else if (index == 7)
		{
			put_color(210, 105, 30); //chocolate
			name = "Chocolate";
		}
		else if (index == 8)
		{
			put_color(0, 0, 0); //black
			name = "Black";
		}
		else if (index == 9)
		{
			put_color(255, 255, 255); //black
			name = "white";
		}
		else
		{
			name = "Err!Overflow";
		}
	}
};
