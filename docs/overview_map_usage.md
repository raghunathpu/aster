# 🗺️ ASTER Overview Map: Usage Guide

The ASTER application now features an interactive **Live Event Density Map** right on the opening **Overview** page. This map provides a geographical representation of historical traffic events across Bengaluru, helping you instantly visualize hotspots and problem areas.

---

## 📍 Navigating the Map

The map is built using PyDeck, which allows for smooth 3D exploration of geospatial data.

*   **Pan:** Click and drag with the left mouse button to move around the map.
*   **Zoom:** Use your mouse scroll wheel, or pinch-to-zoom on a trackpad, to zoom in and out.
*   **Tilt & Rotate (3D View):** Hold down the **Right Mouse Button** (or hold `Ctrl` + Left Click) and drag to tilt the map into 3D and rotate the camera angle. This is particularly useful for seeing overlapping event densities.
*   **Inspect Events:** Hover over any colored point on the map to see a tooltip with the specific details of that event:
    *   **Cause:** The root cause of the traffic event (e.g., Vehicle Breakdown, Accident).
    *   **Corridor:** The road or zone where the event occurred.
    *   **Impact:** The calculated impact tier (Low, Medium, High).

---

## 🎨 Understanding the Data

The points plotted on the map are color-coded based on their operational impact tier, matching ASTER's three-tier system:

| Color | Tier | Operational Meaning |
| :--- | :--- | :--- |
| 🔴 **Red** | **High** | Major disruption. Requires 4-8 officers, full closure, and mandatory diversion. |
| 🟡 **Yellow** | **Medium** | Elevated disruption. Requires 2-4 officers, light barricading, and advisory routing. |
| 🟢 **Green** | **Low** | Routine event. Requires 1-2 officers for monitoring only. |

By exploring this map, you can visually confirm the key operational findings—such as how Accidents and Public Events (often Red/High Impact) cluster along major named corridors, while vehicle breakdowns (often Green/Low Impact or Yellow/Medium Impact) are spread more broadly across the city network.

---

## 🚀 How to Run

If you haven't already started the ASTER Streamlit server, you can do so by running the following command from the project workspace:

```bash
cd aster
streamlit run app/aster_app.py
```

Once the server is running, navigate to the local URL (usually `http://localhost:8501`) and the map will be the first interactive element you see below the top-level KPIs on the **🏠 Overview** page.
