# Game Programming Patterns by Nystorm
-------------------------------------------
* decoupling in software architecture (page 19)
    * large chunk of Design Patterns is about decoupling
    * how is decoupling defined?
        1. <font color=green>Definition 1: </font>if two pieces of code are coupled, it means you can’t understand one without understanding the other. If you de-couple them, you can reason about either side independently
            *  __<font color=blue>A key goal of software architecture: </font>__ minimize the amount of knowledge you need to keep in mind before you can make progress
        2. <font color=green>Definition 2:</font> decoupling is that a change to one piece of code doesn’t necessitate a change to another
* Performance & speed (page 21)
    * ⚠️ Many design patterns for code flexibility rely on virtual dispatch, interfaces, ... that all have at least some runtime cost.
    * spectrum of code flexibility
        * [hard-coded] calling a specific method in some class
        * virtual methods/abstraction
            * the class method that is called isn't known until runtime
        * __template meta-programming__
            * it's somewhere between hard-coded method calls and calling virtual methods
                * you make the decision of which class to call at compile time when the template is instantiated.
            * [DeepSeek] Template metaprogramming (TMP) is a technique in C++ where templates are used to perform computations at compile time rather than runtime. See the example below for computing factorial at compile time NOT run time:
                * ```C++
                    template <int N>
                    struct Factorial {
                        static const int value = N * Factorial<N - 1>::value;
                    };

                    template <>
                    struct Factorial<0> {
                        static const int value = 1;
                    };

                    int main() {
                        int x = Factorial<5>::value; // computed at compile time = 120
                    }
                ```
    * Interesting notes (page 22)
        * observers & messaging
            * observers & messaging let two parts of the game talk to each other which can be extended to three or four
        * game design demands iteration and experimentation.
        * trade-off between code flexiblity & optimization
          * more flexibility $\rightarrow$ faster prototyping, lower performance
          * more code optimization $\rightarrow$ less flexibility, higher performance
* Introduction closing notes
    * Abstraction and decoupling make evolving your program faster and easier, but don’t waste time doing them unless you’re confident the code in question needs that flexibility.
    * Think about & design for performance throughout the development cycle
* until page 27