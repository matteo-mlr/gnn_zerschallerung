if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.complete_graph(15)

    nx.draw_circular(G, with_labels=True, font_weight='bold')
    plt.show()
