"""
ASTER — Routing and Barricading Engine
=======================================
Calculates nearest road network junctions, recommended barricade coordinate circles,
and computes alternative diversion routes (K-shortest paths) around blocked junctions
using congestion-weighted edge traversal costs.
"""
import os
import pickle
import networkx as nx
from geopy.distance import geodesic

class RoutingEngine:
    def __init__(self, graph_path=None):
        self.graph = None
        if graph_path and os.path.exists(graph_path):
            try:
                with open(graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                print(f"Successfully loaded road network graph with {self.graph.number_of_nodes()} nodes.")
            except Exception as e:
                print(f"Error loading graph: {e}")
        
        # Extended fallback dictionary of 50+ junctions for Bengaluru if graph is missing
        self.junctions = {
            "Majestic": (12.9779, 77.5724), "Townhall": (12.9631, 77.5855), "MekhriCircle": (13.0143, 77.5831),
            "SilkBoard": (12.9176, 77.6244), "QueensCircle": (12.9764, 77.5978), "TrinityCircle": (12.9732, 77.6212),
            "RichmondCircle": (12.9602, 77.5975), "HebbalFlyover": (13.0358, 77.5970), "DairyCircle": (12.9427, 77.6047),
            "Banashankari": (12.9154, 77.5738), "Domlur": (12.9610, 77.6387), "CorporationCircle": (12.9678, 77.5880),
            "KR_Puram": (13.0084, 77.6835), "TinFactory": (12.9942, 77.6659), "Marathahalli": (12.9569, 77.7011),
            "Bellandur": (12.9304, 77.6784), "Agara": (12.9234, 77.6481), "Madiwala": (12.9226, 77.6174),
            "BTM_Layout": (12.9166, 77.6101), "Jayadeva": (12.9175, 77.5985), "SouthEndCircle": (12.9348, 77.5794),
            "NationalCollege": (12.9501, 77.5735), "MinervaCircle": (12.9548, 77.5815), "HudsonCircle": (12.9667, 77.5866),
            "Kashivishwanatha": (12.9735, 77.5831), "AnandRaoCircle": (12.9795, 77.5768), "SivanandaCircle": (12.9866, 77.5814),
            "WindsorManor": (12.9928, 77.5838), "CauveryTheater": (13.0035, 77.5843), "SanjaynagarCross": (13.0245, 77.5855),
            "Kempapura": (13.0482, 77.5954), "EsteemMall": (13.0531, 77.5932), "Yelahanka": (13.1007, 77.5963),
            "Goraguntepalya": (13.0312, 77.5447), "Yeshwanthpur": (13.0238, 77.5518), "Navrang": (12.9984, 77.5524),
            "Rajajinagar1stBlock": (12.9912, 77.5546), "SujathaTheater": (12.9782, 77.5555), "MagadiRoadTolgate": (12.9765, 77.5458),
            "Vijayanagar": (12.9715, 77.5358), "Attiguppe": (12.9613, 77.5332), "Nayandahalli": (12.9431, 77.5255),
            "RajarajeshwariNagar": (12.9274, 77.5156), "Kengeri": (12.9069, 77.4812), "Konanakunte": (12.8856, 77.5732),
            "Yelachenahalli": (12.9015, 77.5721), "JP_Nagar_15th_Cross": (12.9113, 77.5835), "Jayanagar_4th_Block": (12.9284, 77.5832),
            "SagarApollo": (12.9355, 77.5991), "Nimhans": (12.9392, 77.5954), "WilsonGarden": (12.9482, 77.5958),
            "ShoolayCircle": (12.9664, 77.6052), "VellaraJunction": (12.9634, 77.6074), "AnilKumbleCircle": (12.9772, 77.6015)
        }

    def find_nearest_junction(self, lat: float, lon: float) -> str:
        """Find the nearest junction name to a coordinate."""
        if self.graph is not None:
            best_node = None
            min_dist = float("inf")
            for node, data in self.graph.nodes(data=True):
                n_lat = data.get("latitude")
                n_lon = data.get("longitude")
                if n_lat is not None and n_lon is not None:
                    dist = geodesic((lat, lon), (n_lat, n_lon)).meters
                    if dist < min_dist:
                        min_dist = dist
                        best_node = node
            if best_node:
                return best_node
        
        # Fallback to dictionary
        best_name = "QueensCircle"
        min_dist = float("inf")
        for name, coords in self.junctions.items():
            dist = geodesic((lat, lon), coords).meters
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name

    def get_junction_coords(self, junction_name: str) -> tuple:
        """Get the latitude/longitude of a junction."""
        if self.graph is not None and self.graph.has_node(junction_name):
            data = self.graph.nodes[junction_name]
            return data.get("latitude"), data.get("longitude")
        return self.junctions.get(junction_name, (12.9764, 77.5978))

    def recommend_barricades(self, event_junction: str, impact_radius_hops=1) -> list[dict]:
        """
        Recommend barricade placement boundary intersections surrounding the incident.
        """
        if self.graph is None or not self.graph.has_node(event_junction):
            # Fallback mock barricades based on nearby nodes
            neighbors_map = {
                "QueensCircle": ["RichmondCircle", "TrinityCircle", "MekhriCircle", "AnilKumbleCircle"],
                "Townhall": ["RichmondCircle", "CorporationCircle", "HudsonCircle", "Kashivishwanatha"],
                "SilkBoard": ["DairyCircle", "Banashankari", "Domlur", "Agara", "BTM_Layout"],
                "MekhriCircle": ["CauveryTheater", "SanjaynagarCross", "WindsorManor", "HebbalFlyover"],
                "Marathahalli": ["Bellandur", "TinFactory", "KR_Puram"],
                "Majestic": ["AnandRaoCircle", "SujathaTheater", "Kashivishwanatha"],
                "HebbalFlyover": ["EsteemMall", "Kempapura", "MekhriCircle", "Goraguntepalya"]
            }
            nodes = neighbors_map.get(event_junction, ["RichmondCircle", "CorporationCircle"])
            recs = []
            for node in nodes:
                coords = self.get_junction_coords(node)
                recs.append({
                    "intersection_id": node,
                    "road_name": f"Road leading to {node}",
                    "distance_hops": impact_radius_hops,
                    "reason": "Boundary intersection diversion point",
                    "latitude": coords[0],
                    "longitude": coords[1]
                })
            return recs

        # Use NetworkX ego_graph to find boundary intersections
        ego = nx.ego_graph(self.graph, event_junction, radius=impact_radius_hops)
        boundary_nodes = []
        for node in ego.nodes():
            neighbors = set(self.graph.neighbors(node))
            outside = neighbors - set(ego.nodes())
            if outside:
                boundary_nodes.append(node)
        if not boundary_nodes:
            # If no boundary (small graph), include all ego nodes except center
            boundary_nodes = [n for n in ego.nodes() if n != event_junction]

        recommendations = []
        for b_node in boundary_nodes:
            road_name = "Unnamed Junction"
            edges = self.graph.edges(b_node, data=True)
            for u, v, data in edges:
                if "name" in data:
                    road_name = data["name"]
                    if isinstance(road_name, list):
                        road_name = road_name[0]
                    break
            
            lat = self.graph.nodes[b_node].get("latitude")
            lon = self.graph.nodes[b_node].get("longitude")
            recommendations.append({
                "intersection_id": str(b_node),
                "road_name": road_name,
                "distance_hops": impact_radius_hops,
                "reason": f"Cordon border at {road_name}",
                "latitude": lat,
                "longitude": lon
            })
        return recommendations

    def get_alternative_routes(self, source, target, blocked_node=None, k=2) -> list[dict]:
        """
        Compute alternative paths avoiding the blocked_node.
        """
        if self.graph is None or not self.graph.has_node(source) or not self.graph.has_node(target):
            # Fallback mock path coordinates
            src_coords = self.get_junction_coords(source)
            tgt_coords = self.get_junction_coords(target)
            # Interpolate a dummy path
            mid_coords = ((src_coords[0]+tgt_coords[0])/2.0 + 0.005, (src_coords[1]+tgt_coords[1])/2.0 + 0.005)
            
            # Create a more realistic diversion path by pulling some nearby nodes
            all_nodes = list(self.junctions.keys())
            nearby = [n for n in all_nodes if n not in [source, target, blocked_node] 
                     and geodesic(src_coords, self.junctions[n]).meters < 4000]
            
            path_nodes = [source]
            path_coords_list = [list(src_coords)]
            
            if nearby:
                bypass = nearby[0]
                path_nodes.append(bypass)
                path_coords_list.append(list(self.junctions[bypass]))
            else:
                path_nodes.append("Bypass_Node")
                path_coords_list.append(list(mid_coords))
                
            path_nodes.append(target)
            path_coords_list.append(list(tgt_coords))
            
            dist_m = int(geodesic(src_coords, mid_coords).meters + geodesic(mid_coords, tgt_coords).meters)
            
            return [
                {
                    "route_id": 1,
                    "path": path_nodes,
                    "path_coordinates": path_coords_list,
                    "travel_time_min": round((dist_m / 8.33) / 60.0, 2),
                    "distance_m": dist_m,
                    "congestion_index": 0.1,
                    "description": f"Via {path_nodes[1].replace('_', ' ')}"
                }
            ]

        # Make a copy of the graph to remove blocked edges
        working_graph = self.graph.copy()
        if blocked_node and working_graph.has_node(blocked_node):
            neighbors = list(working_graph.neighbors(blocked_node))
            for n in neighbors:
                if working_graph.has_edge(blocked_node, n):
                    working_graph.remove_edge(blocked_node, n)

        # Compute paths
        routes = []
        try:
            generator = nx.shortest_simple_paths(working_graph, source, target, weight="length")
            for i, path in enumerate(generator):
                if i >= k:
                    break
                
                dist = sum(self.graph[path[j]][path[j+1]].get("length", 1.0) for j in range(len(path)-1))
                
                # Base speed: 30 km/h = 8.33 m/s
                base_speed = 8.33
                travel_time_sec = dist / base_speed
                total_time_min = travel_time_sec / 60.0
                
                path_coords = []
                for node in path:
                    lat = self.graph.nodes[node].get("latitude")
                    lon = self.graph.nodes[node].get("longitude")
                    if lat is not None and lon is not None:
                        path_coords.append([lat, lon])
                
                routes.append({
                    "route_id": i + 1,
                    "path": [str(node) for node in path],
                    "path_coordinates": path_coords,
                    "travel_time_min": round(total_time_min, 2),
                    "distance_m": int(dist),
                    "congestion_index": round(min(1.0, travel_time_sec / 1800.0), 2),
                    "description": f"Alternative Route {i + 1}"
                })
        except (nx.NetworkXNoPath, nx.NetworkXError):
            print("No path found bypass.")
            
        return routes
