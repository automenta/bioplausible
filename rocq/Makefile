MODULES := Utils.vo EnergyDynamics.vo

all: $(MODULES)

%.vo: %.v Utils.vo
	rocq compile -Q . Computronium $<

Utils.vo: Utils.v
	rocq compile -Q . Computronium $<

clean:
	rm -f *.vo *.vok *.vos *.glob .*.aux

.PHONY: all clean
