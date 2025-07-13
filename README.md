# DeepMine Sentinel AI

An advanced underground mining stope analysis and risk assessment system built with Django.

## 🔧 Issues Fixed

### Critical Issues
1. **✅ Missing Root URL Pattern** - Added home page URL and view
2. **✅ Broken Navigation Links** - Fixed all navbar links with proper URL names
3. **✅ Form Field Issues** - Improved form to exclude auto-generated fields and added proper validation
4. **✅ Static Files Configuration** - Added STATIC_ROOT and STATICFILES_DIRS
5. **✅ Security Issues** - Implemented environment-based configuration for SECRET_KEY and DEBUG
6. **✅ Error Handling** - Added comprehensive error handling in views
7. **✅ Template Context Issues** - Fixed footer template syntax

### Improvements Made
8. **✅ Added `__str__` methods** to all models for better admin interface
9. **✅ Removed unused imports** from settings.py
10. **✅ Created home page view** with dashboard functionality
11. **✅ Added media files configuration** for file uploads
12. **✅ Added pagination** to stope list view
13. **✅ Improved templates** with better styling and user experience
14. **✅ Added logging configuration** for better debugging
15. **✅ Added security settings** for production deployment

## 🚀 Features

- **Dashboard**: Overview of all stopes with statistics
- **Stope Management**: Create, view, and list mining stopes
- **Excel Upload**: Bulk import stopes from Excel files
- **Risk Assessment**: Automated risk analysis based on geological parameters
- **Profile Generation**: AI-powered stope profiling with detailed insights

## 📋 Requirements

- Python 3.8+
- Django 5.2.4
- See `requirements.txt` for complete dependency list

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   cd /path/to/your/project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run migrations**
   ```bash
   python manage.py migrate
   ```

6. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start development server**
   ```bash
   python manage.py runserver
   ```

## 🗂️ Project Structure

```
deepmine_sentinel_ai/
├── core/                           # Main application
│   ├── migrations/                 # Database migrations
│   ├── templates/                  # HTML templates
│   │   ├── core/                   # Base templates
│   │   └── stope/                  # Stope-specific templates
│   ├── static/                     # Static files (CSS, JS, images)
│   ├── models.py                   # Database models
│   ├── views.py                    # View functions
│   ├── forms.py                    # Django forms
│   ├── urls.py                     # URL patterns
│   ├── utils.py                    # Utility functions
│   └── admin.py                    # Admin configuration
├── deepmine_sentinel_ai/           # Project settings
│   ├── settings.py                 # Django settings
│   ├── urls.py                     # Main URL configuration
│   └── wsgi.py                     # WSGI configuration
├── manage.py                       # Django management script
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment variables template
```

## 🎯 Usage

### Creating a Stope
1. Navigate to the home page
2. Click "Add New Stope"
3. Fill in the geological parameters
4. Submit the form
5. View the generated risk assessment and profile

### Bulk Upload
1. Navigate to "Upload Excel"
2. Select an Excel file with the required columns
3. Upload and view the imported stopes

### Required Excel Columns
- `stope_name`: Unique name for the stope
- `rqd`: Rock Quality Designation (%) - number between 0-100
- `hr`: Hydraulic Radius - positive number
- `depth`: Depth below surface (m) - positive number
- `dip`: Dip angle (degrees) - number between 0-90
- `direction`: Direction - must be one of: North, South, East, West, Northeast, Northwest, Southeast, Southwest
- `undercut_wdt`: Undercut width (m) - positive number
- `rock_type`: Rock type - must be one of: Granite, Basalt, Obsidian, Shale, Marble, Slate, Gneiss, Schist, Quartzite, Limestone, Sandstone
- `support_type`: Support type - must be one of: None, Rock Bolts, Mesh, Shotcrete, Timber, Cable Bolts, Steel Sets
- `support_density`: Support density - positive number
- `support_installed`: Boolean (True/False or 1/0)

## 🔒 Security Configuration

### For Production Deployment:

1. **Set environment variables**:
   ```bash
   export DJANGO_SECRET_KEY="your-secret-key-here"
   export DJANGO_DEBUG=False
   export DJANGO_ALLOWED_HOSTS="yourdomain.com,www.yourdomain.com"
   ```

2. **Configure database** (replace SQLite with PostgreSQL/MySQL)

3. **Set up static files serving** with a web server like Nginx

4. **Enable HTTPS** and update security settings

## 🐛 Debugging

- Check `django_errors.log` for error logs
- Use `python manage.py check` to validate configuration
- Run `python manage.py test` to execute tests (when implemented)

## 📊 Database Models

### Stope
- Basic geological and structural parameters
- Support system information
- Timestamps for creation and updates

### StopeProfile
- Generated risk assessment summary
- One-to-one relationship with Stope

### TimeSeriesUpload
- File uploads for time-series data
- Future feature for dynamic analysis

### Prediction
- AI-generated predictions and risk assessments
- Multiple predictions per stope over time

## 🔄 API Endpoints

- `/` - Dashboard home page
- `/create/` - Create new stope
- `/stopes/` - List all stopes
- `/stopes/<id>/` - View specific stope details
- `/upload/` - Excel upload interface
- `/admin/` - Django admin interface

## 🚀 Future Enhancements

- [ ] RESTful API for external integrations
- [ ] Real-time monitoring dashboard
- [ ] Machine learning model integration
- [ ] Advanced visualization charts
- [ ] User authentication and permissions
- [ ] Export functionality
- [ ] Mobile-responsive design improvements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions, please open an issue in the GitHub repository.
