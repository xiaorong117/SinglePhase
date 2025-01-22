.PHONY: build
build: configure
	cmake --build build

.PHONY: configure
configure:
	cmake -B build  -DMUMPS_ROOT=/home/rong/桌面/mumps-main/build/local

.PHONY: run
run:
	./build/myproject

.PHONY: test
test:
	ctest --test-dir build -R "^answer."

.PHONY: clean
clean:
	rm -rf build CMakeFiles main* filename* Permea* myproject Tran* flux* main* out* sub* app* *.vtk
 